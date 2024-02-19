import numpy as np
import torch as th
from einops import rearrange

from .generic_sampler import SimpleWork
from .w_img import split_wimg, avg_merge_wimg

class AvgCircle(SimpleWork):
    def __init__(self, shape, eps_scalar_t_fn, num_img, overlap_size=32):
        c, h, w = shape
        self.base_img_w = w
        self.overlap_size = overlap_size
        self.num_img = num_img
        final_img_w = w * num_img - self.overlap_size * num_img
        super().__init__((c, h, final_img_w), self.get_eps_t_fn(eps_scalar_t_fn))

    def get_eps_t_fn(self, eps_scalar_t_fn):
        def eps_t_fn(long_x, scalar_t, enable_grad=False):
            shift = np.random.randint(self.base_img_w)
            long_x = th.cat(
                [
                    long_x[:,:,:,shift:],
                    long_x[:,:,:,:shift]
                ],
                dim=-1
            )

            x = th.cat(
                [
                    long_x,
                    long_x[:,:,:,:self.overlap_size]
                ],
                dim=-1,
            )
            xs, _overlap = split_wimg(x, self.num_img, rtn_overlap=True)
            assert _overlap == self.overlap_size
            full_eps = eps_scalar_t_fn(xs, scalar_t, enable_grad) # #((b,n), c, h, w)

            eps = avg_merge_wimg(full_eps, self.overlap_size, n=self.num_img)
            eps = th.cat(
                [
                    (eps[:,:,:,:self.overlap_size] + eps[:,:,:,-self.overlap_size:])/2.0,
                    eps[:,:,:,self.overlap_size:-self.overlap_size]
                ],
                dim=-1
            )
            assert eps.shape == long_x.shape
            return th.cat(
                [
                    eps[:,:,:,-shift:],
                    eps[:,:,:,:-shift],
                ],
                dim=-1
            )
            # return eps

        return eps_t_fn

    def x0_fn(self, xt, scalar_t, enable_grad=False):
        cur_eps = self.eps_scalar_t_fn(xt, scalar_t, enable_grad)
        x0 = xt - scalar_t * cur_eps
        return x0, {}, {
            "x0": x0.cpu()
        }