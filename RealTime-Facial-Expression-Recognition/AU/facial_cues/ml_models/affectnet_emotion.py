"""
In-the-wild expression model (EfficientNet-B0, AffectNet 8 classes).

Trained on real faces rather than posed CK+ photos, and it includes a
neutral class. Weights are the public VGAF checkpoint from EmotiEffLib
(Apache-2.0). Reported accuracy is about 61% on AffectNet's 8-class
validation set and about 68% on the VGAF video set.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CLASSES = [
    "anger",
    "contempt",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sadness",
    "surprise",
]

try:
    import torch
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    T = None


class AffectNetEmotion:
    def __init__(self, model_path: str, device: Optional[str] = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for AffectNetEmotion.")
        import timm  # noqa: F401  — needed so torch.load can rebuild the network
        import torch.nn as nn

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Attempt 1: direct torch.load with immediate forward-pass verification
        loaded = False
        try:
            self.model = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.eval()
            dummy = torch.zeros(1, 3, 224, 224, device=self.device)
            with torch.no_grad():
                _ = self.model(dummy)
            loaded = True
        except Exception as direct_exc:
            logger.info("Direct unpickling failed (%s); reconstructing architecture for timm compatibility.", direct_exc)

        # Attempt 2: recreate clean timm model and inject state_dict (handles timm 1.0+ attribute mismatches)
        if not loaded:
            try:
                raw = torch.load(model_path, map_location=self.device, weights_only=False)
                sd = raw.state_dict() if hasattr(raw, "state_dict") else raw
                self.model = timm.create_model("tf_efficientnet_b0_ns", pretrained=False)
                in_features = getattr(self.model.classifier, "in_features", 1280)
                self.model.classifier = nn.Sequential(nn.Linear(in_features, len(CLASSES)))
                self.model.load_state_dict(sd)
                self.model.to(self.device)
                self.model.eval()
                dummy = torch.zeros(1, 3, 224, 224, device=self.device)
                with torch.no_grad():
                    _ = self.model(dummy)
                loaded = True
            except Exception as recon_exc:
                logger.warning("Architecture reconstruction also failed: %s", recon_exc)
                raise

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        logger.info("AffectNet emotion model on %s", self.device)

    def predict(self, img_bgr: np.ndarray, bbox: Optional[tuple] = None) -> Tuple[str, float]:
        face = _crop(img_bgr, bbox)
        if face is None or face.size == 0:
            return "unknown", 0.0
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=-1)[0]
            idx = int(probs.argmax().item())
            conf = float(probs[idx].item())
        return CLASSES[idx], round(conf, 3)


def _crop(img_bgr: np.ndarray, bbox: Optional[tuple]) -> Optional[np.ndarray]:
    if bbox is None:
        return img_bgr
    x, y, w, h = [int(v) for v in bbox]
    pad = int(0.15 * max(w, h))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(img_bgr.shape[1], x + w + pad)
    y1 = min(img_bgr.shape[0], y + h + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    return img_bgr[y0:y1, x0:x1]
