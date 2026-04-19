# visual_localizer.py
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from train_cnn import ResNetEmbedding, verify_match_with_ransac


class VisualLocalizer:
    """
    Maintains a topological position estimate within the prebuilt node map.

    Construction is expensive (one forward pass per node).
    Call update(frame_bgr, img_path) every tick; it returns the confirmed node ID.
    """

    def __init__(
        self,
        model_path: str,        # path to maze_resnet_embedder.pth
        node_img_paths: list,   # ordered list of image file paths matching the node map
        embedding_dim: int = 128,
        top_k: int = 5,         # how many CNN candidates to pass to RANSAC
        cosine_threshold: float = 0.7,   # min cosine sim to consider a candidate
        snap_patience: int = 5,  # how many consecutive misses before forcing a snap
        device: str | None = None,
    ):
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = torch.device(device)

        self.model = ResNetEmbedding(embedding_dim=embedding_dim).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        # ── Precompute node embeddings ────────────────────────────────────────
        # Shape: [N, embedding_dim], L2-normalised (matches training convention)
        self.node_img_paths = node_img_paths
        self.node_embeddings = self._precompute_embeddings(node_img_paths)

        self.current_node: int = 0   # best estimate of where the robot is
        self.off_course: bool = False
        self._miss_streak: int = 0   # consecutive frames with no RANSAC confirmation

        self.top_k = top_k
        self.cosine_threshold = cosine_threshold
        self.snap_patience = snap_patience

    def update(self, frame_bgr, current_frame_path: str) -> int:
        """
        Run one localisation tick.

        Args:
            frame_bgr:           Current camera frame as a numpy BGR array
                                 (as returned by cv2.VideoCapture.read).
            current_frame_path:  Path where this frame is saved on disk.
                                 RANSAC needs to re-read it as a greyscale file.

        Returns:
            Confirmed node ID (int).  Also sets self.off_course = True when
            the robot appears to have drifted away from the expected path.
        """
        query_emb = self._embed_bgr(frame_bgr)                 # [1, D]
        candidates = self._cosine_candidates(query_emb)        # [(score, node_idx), ...]

        confirmed_node = self._ransac_verify(candidates, current_frame_path)

        if confirmed_node is not None:
            self._miss_streak = 0
            self.off_course = False
            self.current_node = confirmed_node
        else:
            self._miss_streak += 1
            if self._miss_streak >= self.snap_patience:
                # Force snap: pick the highest-cosine candidate even without
                # geometric confirmation rather than leaving the planner blind.
                if candidates:
                    self.current_node = candidates[0][1]
                self.off_course = True
                self._miss_streak = 0  # reset so we don't spam snaps
            # else: hold position — keep self.current_node unchanged

        return self.current_node

    def _precompute_embeddings(self, img_paths: list) -> torch.Tensor:
        """
        One-time startup: embed every node image and return a [N, D] tensor.
        This is the map that cosine search runs against every tick.
        """
        print(f"[VisualLocalizer] Precomputing embeddings for {len(img_paths)} nodes…")
        embeddings = []
        with torch.no_grad():
            for path in img_paths:
                img = Image.open(path).convert("RGB")
                tensor = self.transform(img).unsqueeze(0).to(self.device)
                emb = self.model(tensor)                  # already L2-normalised
                embeddings.append(emb)
        result = torch.cat(embeddings, dim=0)             # [N, D]
        print("[VisualLocalizer] Embedding precomputation complete.")
        return result

    def _embed_bgr(self, frame_bgr) -> torch.Tensor:
        """Convert a live BGR numpy frame to an L2-normalised embedding [1, D]."""
        # cv2 gives BGR; PIL and torchvision expect RGB
        rgb = frame_bgr[:, :, ::-1].copy()
        pil = Image.fromarray(rgb)
        tensor = self.transform(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model(tensor)                      # L2-normalised inside model
        return emb                                        # [1, D]

    def _cosine_candidates(self, query_emb: torch.Tensor) -> list[tuple[float, int]]:
        """
        Dot product of an L2-normalised query against all L2-normalised node
        embeddings = cosine similarity.  Returns top-K (score, node_idx) pairs
        above the threshold, sorted descending.
        """
        sims = torch.matmul(query_emb, self.node_embeddings.T).squeeze(0)  # [N]
        scores, indices = torch.topk(sims, k=min(self.top_k, len(self.node_img_paths)))

        candidates = [
            (float(score), int(idx))
            for score, idx in zip(scores, indices)
            if float(score) >= self.cosine_threshold
        ]
        return candidates  # already sorted descending by cosine similarity

    def _ransac_verify(
        self, candidates: list[tuple[float, int]], current_frame_path: str
    ) -> int | None:
        """
        Run geometric verification on candidates in cosine-score order.
        Returns the first node index that passes, or None if all fail.

        RANSAC is intentionally only called on the small top-K shortlist —
        it's 10–100× slower than the CNN step.
        """
        for _score, node_idx in candidates:
            node_path = self.node_img_paths[node_idx]
            if verify_match_with_ransac(current_frame_path, node_path):
                return node_idx
        return None


# Build a localizer from the same artefacts used during training so callers don't have to reconstruct the node list themselves.

def build_localizer_from_training_artefacts(
    model_path: str = "maze_resnet_embedder.pth",
    json_path: str = "data/exploration_data/data_info.json",
    img_dir: str = "data/exploration_data/images",
    subsample_rate: int = 8,
    **kwargs,
) -> VisualLocalizer:
    """
    Mirrors the subsampling logic in train_cnn.py so the node list lines up
    exactly with the topological graph built during preprocessing.
    """
    import json

    with open(json_path, "r") as f:
        data = json.load(f)

    active_steps = [e for e in data if "IDLE" not in e.get("action", [])]
    subsampled = active_steps[::subsample_rate]
    node_img_paths = [os.path.join(img_dir, e["image"]) for e in subsampled]

    return VisualLocalizer(
        model_path=model_path,
        node_img_paths=node_img_paths,
        **kwargs,
    )

def check_facing_forward(current_frame_path, next_node_path):
    """
    Returns True if the robot appears to be facing toward the next node.
    Uses the same RANSAC verification already in visual_localizer.
    """
    return verify_match_with_ransac(current_frame_path, next_node_path,
                                    min_inliers=15,      # looser than localization
                                    min_width_ratio=0.15) # next node is partially visible