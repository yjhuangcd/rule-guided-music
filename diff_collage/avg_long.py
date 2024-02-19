import torch as th
from einops import rearrange

from .generic_sampler import SimpleWork
from .w_img import split_wimg, avg_merge_wimg

class AvgLong(SimpleWork):
    def __init__(self, shape, eps_scalar_t_fn, num_img, overlap_size=32):
        c, h, w = shape
        assert overlap_size == w // 2
        self.overlap_size = overlap_size
        self.num_img = num_img
        final_img_w = w * num_img - self.overlap_size * (num_img - 1)
        super().__init__((c, h, final_img_w), self.get_eps_t_fn(eps_scalar_t_fn))

    def loss(self, x):
        x1, x2 = x[:-1], x[1:]
        return th.sum(
            (th.abs(x1[:, :, :, -self.overlap_size :] - x2[:, :, :, : self.overlap_size])) ** 2,
            dim=(1, 2, 3),
        )

    def get_eps_t_fn(self, eps_scalar_t_fn):
        def eps_t_fn(long_x, scalar_t, y=None):
            xs = split_wimg(long_x, self.num_img, rtn_overlap=False)
            if y is not None:
                y = y.repeat_interleave(self.num_img)
            scalar_t = scalar_t.repeat_interleave(self.num_img)
            full_eps = eps_scalar_t_fn(xs, scalar_t, y=y)  #((b,n), c, h, w)
            full_eps = rearrange(
                full_eps,
                "(b n) c h w -> n b c h w", n = self.num_img
            )

            whole_eps = rearrange(
                full_eps,
                "n b c h w -> (b n) c h w"
            )
            return avg_merge_wimg(whole_eps, self.overlap_size, n=self.num_img, is_avg=False)
        return eps_t_fn