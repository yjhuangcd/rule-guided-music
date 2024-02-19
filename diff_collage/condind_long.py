import torch
import torch as th
from einops import rearrange

from .generic_sampler import SimpleWork
from .w_img import split_wimg, avg_merge_wimg

class CondIndSimple(SimpleWork):
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

            # calculate half eps
            half_eps = eps_scalar_t_fn(xs[:,:,:,-self.overlap_size:], scalar_t, y=y)  #((b,n), c, h, w//2)
            half_eps = rearrange(
                half_eps,
                "(b n) c h w -> n b c h w", n = self.num_img
            )

            half_eps[-1]=0

            full_eps[:,:,:,:,-self.overlap_size:] = full_eps[:,:,:,:,-self.overlap_size:] - half_eps
            whole_eps = rearrange(
                full_eps,
                "n b c h w -> (b n) c h w"
            )
            return avg_merge_wimg(whole_eps, self.overlap_size, n=self.num_img, is_avg=False)
        return eps_t_fn



class CondIndSR(SimpleWork):
    def __init__(self, shape, eps_scalar_t_fn, num_img, low_res, overlap_size=128):
        c, h, w = shape
        assert overlap_size == w // 2
        self.overlap_size = overlap_size
        self.low_overlap_size = low_res.shape[-2] // 2
        self.num_img = num_img
        final_img_w = w * num_img - self.overlap_size * (num_img - 1)
        assert low_res.shape[-1] == self.low_overlap_size * (num_img + 1)

        self.square_fn = self.get_square_sr_fn(eps_scalar_t_fn, low_res)
        self.half_fn = self.get_half_sr_fn(eps_scalar_t_fn, low_res)

        super().__init__((c, h, final_img_w), self.get_eps_t_fn())

    def get_square_sr_fn(self, eps_fn, low_res):
        low_res = split_wimg(low_res, self.num_img, False)
        def _fn(_x, _t, enable_grad):
            context = th.enable_grad if enable_grad else th.no_grad
            with context():
                vec_t = th.ones(_x.shape[0]).cuda() * _t
                rtn = eps_fn(_x, vec_t, low_res)
            rtn = rearrange(
                rtn,
                "(b n) c h w -> n b c h w", n = self.num_img
            )
            return rtn
        return _fn

    def get_half_sr_fn(self, eps_fn, low_res):
        low_res = split_wimg(low_res, self.num_img, False)
        def _fn(_x, _t, enable_grad):
            context = th.enable_grad if enable_grad else th.no_grad
            with context():
                vec_t = th.ones(_x.shape[0]).cuda() * _t
                half_eps = eps_fn(_x[:,:,:,-self.overlap_size:], vec_t, low_res[:,:,:,-self.low_overlap_size:])
            half_eps = rearrange(
                half_eps,
                "(b n) c h w -> n b c h w", n = self.num_img
            )

            half_eps[-1]=0
            return half_eps
        return _fn

    def get_eps_t_fn(self):
        def eps_t_fn(in_x, scalar_t, enable_grad=False):
            xs = split_wimg(in_x, self.num_img, rtn_overlap=False)

            # full eps
            full_eps = self.square_fn(xs, scalar_t, enable_grad)
            # calculate half eps
            half_eps = self.half_fn(xs, scalar_t, enable_grad)

            full_eps[:,:,:,:,-self.overlap_size:] = full_eps[:,:,:,:,-self.overlap_size:] - half_eps
            whole_eps = rearrange(
                full_eps,
                "n b c h w -> (b n) c h w"
            )
            out_eps = avg_merge_wimg(whole_eps, self.overlap_size, n=self.num_img, is_avg=False)
            return out_eps
        return eps_t_fn




# class CondIndLong(SimpleWork):
#     def __init__(self, shape, eps_scalar_t_fn, overlap_size=32):
#         super().__init__(shape, eps_scalar_t_fn)
#         self.overlap_size = overlap_size

#     def loss(self, x):
#         x1, x2 = x[:-1], x[1:]
#         return th.sum(
#             (th.abs(x1[:, :, :, -self.overlap_size :] - x2[:, :, :, : self.overlap_size])) ** 2,
#             dim=(1, 2, 3),
#         )

#     def generate_xT(self, n):
#         white_noise = th.randn((n , *self.shape)).cuda()
#         return self.noise(white_noise, None) * 80.0

#     def noise(self, xt, scalar_t):
#         del scalar_t
#         noise = th.randn_like(xt)
#         b, _, _, w = xt.shape
#         final_img_w = w * b - self.overlap_size * (b - 1)
#         noise = rearrange(noise, "(t n) c h w -> t c h (n w)", t=1)[:, :, :, :final_img_w]
#         noise = split_wimg(noise, b, rtn_overlap=False)
#         return noise

#     def merge(self, xs):
#         return avg_merge_wimg(xs, self.overlap_size)