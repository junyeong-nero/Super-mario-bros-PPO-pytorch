import numpy as np
import torch
from PIL import Image
from typing import Union

# Try to import LazyFrames if gym is installed, otherwise handle gracefully
try:
    from gym.wrappers.frame_stack import LazyFrames
except ImportError:
    LazyFrames = None

# Define a custom type for input images
InputImageType = Union[np.ndarray, Image.Image, "LazyFrames"]


def to_pil_image(image: InputImageType) -> Image.Image:
    """
    Converts various input types (LazyFrames, ndarray) into a PIL Image.

    Args:
        image: Input image as LazyFrames, numpy array, or PIL Image.

    Returns:
        PIL.Image: The converted image.

    Raises:
        ValueError: If the numpy array shape is not supported.
        TypeError: If the input type is not supported.
    """
    # 1. Handle LazyFrames (Gym wrappers)
    if LazyFrames is not None and isinstance(image, LazyFrames):
        image = np.array(image)

    # 2. Handle Numpy Arrays
    if isinstance(image, np.ndarray):
        # Remove empty dimensions (e.g., 1x84x84 -> 84x84)
        image = np.squeeze(image)

        # Scale float images (0.0-1.0) to int (0-255) if necessary
        if image.dtype != np.uint8:
            image = (255 * image).clip(0, 255).astype(np.uint8)

        # Determine mode based on dimensions
        if image.ndim == 2:
            mode = "L"  # Grayscale
        elif image.ndim == 3 and image.shape[2] == 3:
            mode = "RGB"  # Color
        else:
            raise ValueError(f"Unsupported ndarray shape: {image.shape}")

        return Image.fromarray(image, mode)

    # 3. Handle PIL Images (Pass through)
    if isinstance(image, Image.Image):
        return image

    raise TypeError(f"Unsupported image type: {type(image)}")


def preprocess_image(img: InputImageType, device: torch.device = None) -> torch.Tensor:
    """
    Preprocesses an image for the network:
    1. Converts to PIL Image.
    2. Resizes to 84x84.
    3. Converts to Grayscale.
    4. Normalizes to 0-1 float32.
    5. Converts to PyTorch Tensor with batch and channel dims.

    Args:
        img: Input image (PIL, LazyFrame, or Numpy).
        device: Target device for the tensor (cpu, cuda, mps).

    Returns:
        torch.Tensor: Preprocessed tensor with shape (1, 1, 84, 84).
    """
    # Ensure input is a PIL image
    img = to_pil_image(img)

    # Resize (84x84) and Convert to Grayscale ('L')
    img = img.resize((84, 84), Image.BILINEAR).convert("L")

    # Convert to numpy and normalize (0 to 1)
    arr = np.array(img, dtype=np.float32) / 255.0

    # Convert to Tensor
    tensor = torch.from_numpy(arr)

    # Add Batch and Channel dimensions: (H, W) -> (1, 1, H, W)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    # Move to device if specified
    if device is not None:
        tensor = tensor.to(device)

    return tensor
