import torch as th
from einops import rearrange

from .generic_sampler import SimpleWork
from .w_img import split_wimg, avg_merge_wimg

class CondIndCircle(SimpleWork):
    def __init__(self, shape, eps_scalar_t_fn, num_img, overlap_size=32):
        c, h, w = shape
        assert overlap_size == w // 2
        self.overlap_size = overlap_size
        self.num_img = num_img
        final_img_w = w * num_img - self.overlap_size * num_img
        super().__init__((c, h, final_img_w), self.get_eps_t_fn(eps_scalar_t_fn))

    def circle_split(self, in_x):
        long_x = th.cat(
            [
                in_x,
                in_x[:,:,:,:self.overlap_size],
            ],
            dim=-1
        )
        xs = split_wimg(long_x, self.num_img, rtn_overlap=False)
        return xs

    def circle_merge(self, xs, overlap_size=None):
        if overlap_size is None:
            overlap_size = self.overlap_size
        long_xs = avg_merge_wimg(xs, overlap_size, n=self.num_img, is_avg=True)
        return th.cat(
            [
                (
                    long_xs[:,:,:,:overlap_size] + long_xs[:,:,:,-overlap_size:]
                ) / 2.0,
                long_xs[:,:,:,overlap_size:-overlap_size]
            ],
            dim=-1
        )

    def get_eps_t_fn(self, eps_scalar_t_fn):
        def eps_t_fn(in_x, scalar_t, y=None):
            long_x = th.cat(
                [
                    in_x,
                    in_x[:,:,:,:self.overlap_size],
                ],
                dim=-1
            )
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
            half_eps = eps_scalar_t_fn(xs[:,:,:,-self.overlap_size:], scalar_t, y=y) #((b,n), c, h, w//2)
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
            long_eps = avg_merge_wimg(whole_eps, self.overlap_size, n=self.num_img, is_avg=False)
            return th.cat(
                [
                    (
                        long_eps[:,:,:,:self.overlap_size] + long_eps[:,:,:,-self.overlap_size:]
                    ) / 2.0,
                    long_eps[:,:,:,self.overlap_size:-self.overlap_size]
                ],
                dim=-1
            )
        return eps_t_fn


class CondIndCircleSR(SimpleWork):
    def __init__(self, shape, eps_scalar_t_fn, num_img, low_res, overlap_size=32):
        c, h, w = shape
        assert overlap_size == w // 2
        self.overlap_size = overlap_size
        self.low_overlap_size = low_res.shape[-2] // 2
        self.num_img = num_img
        final_img_w = w * num_img - self.overlap_size * num_img
        assert low_res.shape[-1] == self.low_overlap_size * num_img

        self.square_fn = self.get_square_sr_fn(eps_scalar_t_fn, low_res)
        self.half_fn = self.get_half_sr_fn(eps_scalar_t_fn, low_res)

        super().__init__((c, h, final_img_w), self.get_eps_t_fn())

    def circle_split(self, in_x, overlap_size=None):
        if overlap_size is None:
            overlap_size = self.overlap_size
        long_x = th.cat(
            [
                in_x,
                in_x[:,:,:,:overlap_size],
            ],
            dim=-1
        )
        xs = split_wimg(long_x, self.num_img, rtn_overlap=False)
        return xs

    def circle_merge(self, xs, overlap_size=None):
        if overlap_size is None:
            overlap_size = self.overlap_size
        long_xs = avg_merge_wimg(xs, overlap_size, n=self.num_img, is_avg=True)
        return th.cat(
            [
                (
                    long_xs[:,:,:,:overlap_size] + long_xs[:,:,:,-overlap_size:]
                ) / 2.0,
                long_xs[:,:,:,overlap_size:-overlap_size]
            ],
            dim=-1
        )

    def get_square_sr_fn(self, eps_fn, low_res):
        low_res = self.circle_split(low_res, self.low_overlap_size)
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
        low_res = self.circle_split(low_res, self.low_overlap_size)
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
            long_x = th.cat(
                [
                    in_x,
                    in_x[:,:,:,:self.overlap_size],
                ],
                dim=-1
            )
            xs = split_wimg(long_x, self.num_img, rtn_overlap=False)

            # full eps
            full_eps = self.square_fn(xs, scalar_t, enable_grad)
            # calculate half eps
            half_eps = self.half_fn(xs, scalar_t, enable_grad)

            full_eps[:,:,:,:,-self.overlap_size:] = full_eps[:,:,:,:,-self.overlap_size:] - half_eps
            whole_eps = rearrange(
                full_eps,
                "n b c h w -> (b n) c h w"
            )
            long_eps = avg_merge_wimg(whole_eps, self.overlap_size, n=self.num_img, is_avg=False)
            return th.cat(
                [
                    (
                        long_eps[:,:,:,:self.overlap_size] + long_eps[:,:,:,-self.overlap_size:]
                    ) / 2.0,
                    long_eps[:,:,:,self.overlap_size:-self.overlap_size]
                ],
                dim=-1
            )
        return eps_t_fn