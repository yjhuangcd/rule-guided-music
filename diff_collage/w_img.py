import torch as th
from einops import rearrange

__all__ = [
    "split_wimg",
]

def split_wimg(wimg, n_img, rtn_overlap=True):
    if wimg.ndim == 3:
        wimg = wimg[None]
    _, _, h, w = wimg.shape
    base_len = 128   # todo: hard code 128 here (the length of the latents)
    overlap_size = (n_img * base_len - w) // (n_img - 1)
    assert n_img * base_len - overlap_size * (n_img - 1) == w
    
    img = th.nn.functional.unfold(wimg, kernel_size=(h, base_len), stride=base_len - overlap_size) #(B, block, n_img)
    img = rearrange(
        img,
        "b (c h w) n -> (b n) c h w", h=h, w=base_len
    )
    
    if rtn_overlap:
        return img , overlap_size
    return img

def avg_merge_wimg(imgs, overlap_size, n=None, is_avg=True):
    b, _, h, w = imgs.shape
    if n == None:
        n = b
    unfold_img = rearrange(
        imgs,
        "(b n) c h w -> b (c h w) n", n = n
    )
    img = th.nn.functional.fold(
        unfold_img,
        (h, n * w - (n-1) * overlap_size),
        kernel_size = (h, w),
        stride = w - overlap_size
    ) 
    if is_avg:
        counter = th.nn.functional.fold(
            th.ones_like(unfold_img), 
            (h, n * w - (n-1) * overlap_size),
            kernel_size = (h, w),
            stride = w - overlap_size
        )
        return img / counter
    return img

# legacy code use naive implementation

def split_wimg_legacy(himg, n_img, rtn_overlap=True):
    if himg.ndim == 3:
        himg = himg[None]
    _, _, h, w = himg.shape
    overlap_size = (n_img * h - w) // (n_img - 1)
    assert n_img * h - overlap_size * (n_img - 1) == w
    himg = himg[0]
    rtn_img = [himg[:, :, :h]]
    for i in range(n_img - 1):
        rtn_img.append(himg[:, :, (h - overlap_size) * (i + 1) : h + (h - overlap_size) * (i + 1)])
    if rtn_overlap:
        return th.stack(rtn_img), overlap_size
    return th.stack(rtn_img)

def avg_merge_wimg_legacy(imgs, overlap_size):
    _, _, _, w = imgs.shape
    rtn_img = [imgs[0]]
    for cur_img in imgs[1:]:
        rtn_img.append(cur_img[:, :, overlap_size:])
    first_img = th.cat(rtn_img, dim=-1)

    rtn_img = []
    for cur_img in imgs[:-1]:
        rtn_img.append(cur_img[:, :, : w - overlap_size])
    rtn_img.append(imgs[-1])
    second_img = th.cat(rtn_img, dim=-1)

    return (first_img + second_img) / 2.0
