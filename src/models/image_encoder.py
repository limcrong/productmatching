from pathlib import Path
from typing import List, Optional, Union
from PIL import Image
import numpy as np

try:
    import torch
    import timm
except ImportError:
    torch = None
    timm = None

from src.config import DEFAULT_IMAGE_MODEL
from src.data.preprocessing import get_image_transforms


class ImageEncoder:
    """
    Deep learning visual feature extractor using timm (PyTorch Image Models).
    Extracts L2-normalized feature vectors from product images.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_IMAGE_MODEL,
        pretrained: bool = True,
        device: Optional[str] = None,
    ):
        if torch is None or timm is None:
            raise ImportError("torch and timm are required for ImageEncoder")

        self.model_name = model_name
        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # num_classes=0 removes final classification layer and outputs pooled embeddings
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.model.to(self.device)
        self.model.eval()

        self.transforms = get_image_transforms()

    @torch.no_grad()
    def encode_images(
        self,
        image_paths: List[Union[str, Path]],
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Processes a list of image paths and returns an array of embeddings (N, D).
        """
        all_embeddings = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch_tensors = []

            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    tensor = self.transforms(img)
                except Exception:
                    # In case of corrupted or missing image, use a zero placeholder
                    tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
                batch_tensors.append(tensor)

            stacked = torch.stack(batch_tensors).to(self.device)
            feats = self.model(stacked)

            if normalize:
                feats = torch.nn.functional.normalize(feats, p=2, dim=1)

            all_embeddings.append(feats.cpu().numpy())

        if not all_embeddings:
            return np.empty((0, 0), dtype=np.float32)

        return np.vstack(all_embeddings).astype(np.float32)
