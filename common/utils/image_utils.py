from typing import Sequence, List

import torch
import numpy as np
from einops import rearrange


def image_to_torch(image: np.ndarray) -> torch.Tensor:
    """
    Transpose an image or batch of images (re-order channels).

    :param image:
    :return:
    """
    image = torch.as_tensor(rearrange(image, "... h w c -> ... c h w"))
    if image.dtype == torch.uint8:
        return image.float() / 255
    return image.float()


def torch_to_image(image: torch.Tensor, transpose=True) -> np.ndarray:
    """
    Transpose an image or batch of images (re-order channels).

    :param image:
    :return:
    """
    if transpose:
        image = rearrange(image, "... c h w -> ... h w c")
    image = image.cpu().data.numpy()
    image = np.clip(image, 0, 1) * 255
    return image.astype(np.uint8)


def tile_images(img_nhwc: Sequence[np.ndarray]) -> np.ndarray:  # pragma: no cover
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.

    :param img_nhwc: list or array of images, ndim=4 once turned into array. img nhwc
        n = batch index, h = height, w = width, c = channel
    :return: img_HWc, ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    n_images, height, width, n_channels = img_nhwc.shape
    # new_height was named H before
    new_height = int(np.ceil(np.sqrt(n_images)))
    # new_width was named W before
    new_width = int(np.ceil(float(n_images) / new_height))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(n_images, new_height * new_width)])
    # img_HWhwc
    out_image = img_nhwc.reshape((new_height, new_width, height, width, n_channels))
    # img_HhWwc
    out_image = out_image.transpose(0, 2, 1, 3, 4)
    # img_Hh_Ww_c
    out_image = out_image.reshape((new_height * height, new_width * width, n_channels))
    return out_image


def save_video_from_images(images: List[np.ndarray], save_path, fps=30):
    import moviepy.editor as mpy
    clip = mpy.ImageSequenceClip(images, fps=fps)
    clip.write_videofile(save_path, fps=fps, codec='mpeg4', logger=None)
