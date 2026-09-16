import re
import html
from typing import Optional
from PIL import Image


def clean_text(text: str) -> str:
    """
    Cleans raw e-commerce product titles:
    - HTML unescaping
    - Lowercasing
    - Normalizing whitespace and noisy special characters
    """
    if not isinstance(text, str):
        return ""
    
    # Decode html entities like &amp;, &quot;
    text = html.unescape(text)
    
    # Lowercase
    text = text.lower()
    
    # Remove URL links if present
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    
    # Replace non-alphanumeric characters (retaining basic symbols if needed)
    text = re.sub(r"[^\w\s\-\.]", " ", text)
    
    # Compress multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def get_image_transforms(img_size: int = 224):
    """
    Standard evaluation transform for computer vision models (e.g. timm / EfficientNet / ViT).
    """
    try:
        from torchvision import transforms
    except ImportError:
        raise ImportError("torchvision is required for image transforms")
        
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
