import numpy as np
from gym.wrappers.frame_stack import LazyFrames
from PIL import Image


def to_pil_image(image):
    # unwrap LazyFrames
    if LazyFrames is not None and isinstance(image, LazyFrames):
        image = np.array(image)

    # handle ndarray
    if isinstance(image, np.ndarray):
        image = np.squeeze(image)

        if image.dtype != np.uint8:
            image = (255 * image).clip(0, 255).astype(np.uint8)

        if image.ndim == 2:
            mode = "L"
        elif image.ndim == 3 and image.shape[2] == 3:
            mode = "RGB"
        else:
            raise ValueError(f"Unsupported ndarray shape: {image.shape}")

        return Image.fromarray(image, mode)

    # If it's already a PIL image
    if isinstance(image, Image.Image):
        return image

    raise TypeError(f"Unsupported image type: {type(image)}")


import torch
import numpy as np
from PIL import Image


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """
    PyTorch Tensor (C, H, W) → PIL.Image 변환 함수
    """
    if t.is_cuda:
        t = t.cpu()

    t = t.squeeze(0)
    print(t.shape)

    # 값 범위가 0~1이면 0~255로 변환
    if t.max() <= 1:
        t = t * 255

    # (C, H, W) → (H, W, C)
    t = t.clamp(0, 255).byte().permute(1, 2, 0)

    return Image.fromarray(t.numpy())
