from pathlib import Path
from typing import List, Optional, Union, Tuple
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
        images_or_paths: List[Union[str, Path, Image.Image, None]],
        batch_size: int = 32,
        normalize: bool = True,
        return_mask: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Processes a list of image paths or PIL Image instances and returns an array of embeddings (N, D).
        If return_mask is True, also returns a boolean array indicating valid (non-corrupted) images.
        """
        all_embeddings = []
        all_masks = []

        for i in range(0, len(images_or_paths), batch_size):
            batch_items = images_or_paths[i : i + batch_size]
            batch_tensors = []
            batch_mask = []

            for item in batch_items:
                tensor = None
                is_valid = False
                if item is not None:
                    try:
                        if isinstance(item, Image.Image):
                            img = item.convert("RGB")
                        else:
                            img = Image.open(item).convert("RGB")
                        tensor = self.transforms(img)
                        is_valid = True
                    except Exception:
                        tensor = None
                        is_valid = False

                if tensor is None:
                    tensor = torch.zeros((3, 224, 224), dtype=torch.float32)

                batch_tensors.append(tensor)
                batch_mask.append(is_valid)

            stacked = torch.stack(batch_tensors).to(self.device)
            feats = self.model(stacked)

            if normalize:
                feats = torch.nn.functional.normalize(feats, p=2, dim=1)

            all_embeddings.append(feats.cpu().numpy())
            all_masks.extend(batch_mask)

        if not all_embeddings:
            empty_emb = np.empty((0, 0), dtype=np.float32)
            empty_mask = np.empty((0,), dtype=bool)
            return (empty_emb, empty_mask) if return_mask else empty_emb

        embeddings = np.vstack(all_embeddings).astype(np.float32)
        valid_mask = np.array(all_masks, dtype=bool)

        return (embeddings, valid_mask) if return_mask else embeddings

