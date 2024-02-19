import math
import torch as th
from einops import rearrange
import numpy as np

from .generic_sampler import batch_mul


def split_wimg(himg, n_img, rtn_overlap=True):
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


def merge_wimg(imgs, overlap_size):
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


def get_x0_pred_fn(raw_net_model, cond_loss_fn, weight_fn, x0_fn, thres_t, init_fn=None):
    def fn(xt, scalar_t):
        if init_fn is not None:
            xt = init_fn(xt, scalar_t)
        xt = xt.requires_grad_(True)
        x0_pred = raw_net_model(xt, scalar_t)

        loss_info = {
            "raw_x0": cond_loss_fn(x0_pred.detach()).cpu(),
        }
        traj_info = {
            "t": scalar_t,
        }
        if scalar_t < thres_t:
            x0_cor = x0_pred.detach()
        else:
            pred_loss = cond_loss_fn(x0_pred)
            grad_term = th.autograd.grad(pred_loss.sum(), xt)[0]
            weights = weight_fn(x0_pred, grad_term, cond_loss_fn)
            x0_cor = (x0_pred - batch_mul(weights, grad_term)).detach()
            loss_info["weight"] = weights.detach().cpu()
            traj_info["grad"] = grad_term.detach().cpu()

        if x0_fn:
            x0 = x0_fn(x0_cor, scalar_t)
        else:
            x0 = x0_cor

        loss_info["cor_x0"] = cond_loss_fn(x0_cor.detach()).cpu()
        loss_info["x0"] = cond_loss_fn(x0.detach()).cpu()
        traj_info.update({
                "raw_x0": x0_pred.detach().cpu(),
                "cor_x0": x0_cor.detach().cpu(),
                "x0": x0.detach().cpu(),
            }
        )
        return x0_cor, loss_info, traj_info

    return fn


def simple_noise(cur_t, xt):
    del cur_t
    return th.randn_like(xt)


def get_fix_weight_fn(fix_weight):
    def weight_fn(xs, grads, *args):
        del grads, args
        return th.ones(xs.shape[0]).to(xs) * fix_weight

    return weight_fn


class SeqWorker:
    def __init__(self, overlap_size=10, src_img=None):
        self.overlap_size = overlap_size
        self.src_img = src_img

    def loss(self, x):
        return th.sum(
            (th.abs(self.src_img[:, :, :, -self.overlap_size :] - x[:, :, :, : self.overlap_size]))
            ** 2,
            dim=(1, 2, 3),
        )

    def x0_replace(self, x0):
        rtn_x0 = x0.clone()
        rtn_x0[:, :, :, : self.overlap_size] = self.src_img[:, :, :, -self.overlap_size :]
        return x0

    def optimal_weight_fn(self, x0, grads, *args, ratio=1.0):
        del args
        overlap_size = self.overlap_size
        # argmin_{w} (delta_pixel - w * delta_pixel)^2
        delta_pixel = x0[:, :, :, :overlap_size] - self.src_img[:, :, :, -overlap_size:]
        delta_grads = grads[:, :, :, :overlap_size]
        num = th.sum(delta_pixel * delta_grads).item()
        denum = th.sum(delta_grads * delta_grads).item()
        _optimal_weight = num / denum
        if math.isnan(_optimal_weight):
            print(denum)
            raise RuntimeError("nan for weights")

        return ratio * _optimal_weight * th.ones(x0.shape[0]).to(x0)


class CircleWorker:
    def __init__(self, overlap_size=10, adam_num_iter=100):
        self.overlap_size = overlap_size
        self.adam_num_iter = adam_num_iter


    def get_match_patch(self, x):
        tail = x[:, :, :, -self.overlap_size :]
        head = x[:, :, :, : self.overlap_size]
        tail = th.roll(tail, 1, 0)
        return tail, head

    def loss(self, x):
        tail, head = self.get_match_patch(x)
        return th.sum(
            (tail - head)**2,
            dim=(1, 2, 3),
        )

    def split_noise(self, cur_t, xt):
        noise = simple_noise(cur_t, xt)
        b, _, _, w = xt.shape
        final_img_w = w * b - self.overlap_size * b
        noise = rearrange(noise, "(t n) c h w -> t c h (n w)", t=1)[:, :, :, :final_img_w]
        noise = th.cat([noise, noise[:,:,:, :self.overlap_size]], dim=-1)
        noise, _ = split_wimg(noise, b)
        return noise

    def merge_circle_image(self, xt):
        merged_long_img = merge_wimg(xt, self.overlap_size)
        return th.cat(
            [
                (merged_long_img[:,:,:self.overlap_size] + merged_long_img[:,:,-self.overlap_size:]) / 2.0,
                merged_long_img[:,:,self.overlap_size:-self.overlap_size],
            ],
            dim=-1
        )

    def split_circle_image(self, merged_long_img, n):
        imgs,_ = split_wimg(
            th.cat(
                [
                    merged_long_img,
                    merged_long_img[:,:,:self.overlap_size],
                ],
                dim = -1,
            ),
            n
        )
        return imgs


    def optimal_weight_fn(self, xs, grads, *args):
        del args
        # argmin_{w} (delta_pixel - w * delta_pixel)^2
        tail, head = self.get_match_patch(xs)
        delta_pixel = tail - head
        tail, head = self.get_match_patch(grads)
        delta_grads = tail - head

        num = th.sum(delta_pixel * delta_grads).item()
        denum = th.sum(delta_grads * delta_grads).item()
        _optimal_weight = num / denum
        return _optimal_weight * th.ones(xs.shape[0]).to(xs)

    def adam_grad_weight(self, x0, grad_term, cond_loss_fn):
        init_weight = self.optimal_weight_fn(x0, grad_term)
        grad_term = grad_term.detach()
        x0 = x0.detach()
        with th.enable_grad():
            weights = init_weight.requires_grad_()
            optimizer = th.optim.Adam(
                [
                    weights,
                ],
                lr=1e-2,
            )

            def _loss(w):
                cor_x0 = x0 - batch_mul(w, grad_term)
                return cond_loss_fn(cor_x0).sum()

            for _ in range(self.adam_num_iter):
                optimizer.zero_grad()
                _cur_loss = _loss(weights)
                _cur_loss.backward()
                optimizer.step()
        return weights

    # TODO:
    def x0_replace(self, x0, sclar_t, thres_t):
        if sclar_t > thres_t:
            merge_x0 = merge_wimg(x0, self.overlap_size)
            return split_wimg(merge_x0, x0.shape[0])[0]
        else:
            return x0


class ParaWorker:
    def __init__(self, overlap_size=10, adam_num_iter=100):
        self.overlap_size = overlap_size
        self.adam_num_iter = adam_num_iter

    def loss(self, x):
        x1, x2 = x[:-1], x[1:]
        return th.sum(
            (th.abs(x1[:, :, :, -self.overlap_size :] - x2[:, :, :, : self.overlap_size])) ** 2,
            dim=(1, 2, 3),
        )

    def split_noise(self, xt, cur_t):
        noise = simple_noise(cur_t, xt)
        b, _, _, w = xt.shape
        final_img_w = w * b - self.overlap_size * (b - 1)
        noise = rearrange(noise, "(t n) c h w -> t c h (n w)", t=1)[:, :, :, :final_img_w]
        noise, _ = split_wimg(noise, b)
        return noise

    def optimal_weight_fn(self, xs, grads, *args):
        del args
        overlap_size = self.overlap_size
        # argmin_{w} (delta_pixel - w * delta_pixel)^2
        delta_pixel = xs[:-1, :, :, -overlap_size:] - xs[1:, :, :, :overlap_size]
        delta_grads = grads[:-1, :, :, -overlap_size:] - grads[1:, :, :, :overlap_size]
        num = th.sum(delta_pixel * delta_grads).item()
        denum = th.sum(delta_grads * delta_grads).item()
        _optimal_weight = num / denum
        return _optimal_weight * th.ones(xs.shape[0]).to(xs)

    def adam_grad_weight(self, x0, grad_term, cond_loss_fn):
        init_weight = self.optimal_weight_fn(x0, grad_term)
        grad_term = grad_term.detach()
        x0 = x0.detach()
        with th.enable_grad():
            weights = init_weight.requires_grad_()
            optimizer = th.optim.Adam(
                [
                    weights,
                ],
                lr=1e-2,
            )

            def _loss(w):
                cor_x0 = x0 - batch_mul(w, grad_term)
                return cond_loss_fn(cor_x0).sum()

            for _ in range(self.adam_num_iter):
                optimizer.zero_grad()
                _cur_loss = _loss(weights)
                _cur_loss.backward()
                optimizer.step()
        return weights

    def x0_replace(self, x0, sclar_t, thres_t):
        if sclar_t > thres_t:
            merge_x0 = merge_wimg(x0, self.overlap_size)
            return split_wimg(merge_x0, x0.shape[0])[0]
        else:
            return x0

class ParaWorkerC(ParaWorker):
    def __init__(self, src_img, mask_img, inpaint_w = 1.0, overlap_size=10, adam_num_iter=100):
        self.src_img = src_img
        self.inpaint_w = inpaint_w
        self.mask_img = mask_img # 1 indicate masked given pixels
        super().__init__(overlap_size, adam_num_iter)

    def loss(self, x):
        if x.shape[0] == 1:
            return th.sum(
                th.sum(
                    th.square(self.src_img[:,:,:,:x.shape[-1]] - x), dim=(0,1)
                ) * self.mask_img[:,:x.shape[-1]]
            )
        else:
            consistent_loss = super().loss(x)
            # merge image
            merge_x = merge_wimg(x, self.overlap_size)

            inpating_loss = th.sum(
                th.sum(
                    th.square(self.src_img[:,:,:,:merge_x.shape[-1]] - merge_x), dim=(0,1)
                ) * self.mask_img[:,:merge_x.shape[-1]]
            )

        return consistent_loss + inpating_loss / (x.shape[-1] - 1)

    def x0_replace(self, x0, sclar_t, thres_t):
        if sclar_t > thres_t:
            merge_x = merge_wimg(x0, self.overlap_size)
            src_img = self.src_img[:,:,:,:merge_x.shape[-1]]
            mask_img = self.mask_img[:,:merge_x.shape[-1]]
            merge_x = th.where(mask_img[None,None], src_img, merge_x)
            return split_wimg(merge_x, x0.shape[0])[0]
        else:
            return x0


class SplitMergeOp:
    def __init__(self, avg_overlap=32):
        self.avg_overlap = avg_overlap
        self.cur_overlap_int = None

    def sample(self, n):
        # lower_coef = 3 / 4.0
        _lower_bound = self.avg_overlap - 6
        base_overlap = np.ones(n) * _lower_bound

        total_ball = (self.avg_overlap - _lower_bound) * n
        random_number = np.random.randint(0, total_ball - n, n-1)
        random_number = np.sort(random_number)
        balls = np.append(random_number, total_ball - n) - np.insert(random_number, 0, 0) + np.ones(n) + base_overlap

        assert np.sum(balls) == n * self.avg_overlap

        # TODO: FIXME
        balls = np.ones(n) * self.avg_overlap

        return balls.astype(np.int)

    def reset(self, n):
        self.cur_overlap_int = self.sample(n)

    def split(self, img, n, img_w=64):
        assert img.ndim == 3
        # assert img.shape[-1] > (n-1) * self.avg_overlap
        assert (n-1) == self.cur_overlap_int.shape[0]

        assert (n-1) * self.avg_overlap + img.shape[-1] == n * img_w

        cur_idx = 0
        imgs = []
        for cur_overlap in self.cur_overlap_int:
            imgs.append(img[:,:,cur_idx:cur_idx + img_w])
            cur_idx = cur_idx + img_w - cur_overlap
        imgs.append(img[:,:,cur_idx:])
        return th.stack(imgs)

    def merge(self, imgs):
        b = imgs.shape[0]
        img_size = imgs.shape[-1]
        assert b - 1 == self.cur_overlap_int.shape[0]
        img_width = b * imgs.shape[-1] - np.sum(self.cur_overlap_int)
        wimg = th.zeros((3, imgs.shape[-2], img_width)).to(imgs)
        ncnt = th.zeros(img_width).to(imgs)
        cur_idx = 0
        for i_th, cur_img in enumerate(imgs):
            wimg[:,:,cur_idx:cur_idx + img_size] += cur_img
            ncnt[cur_idx:cur_idx + img_size] += 1.0
            if i_th < b -1:
                cur_idx = cur_idx + img_size - self.cur_overlap_int[i_th]
        return wimg / ncnt[None,None,:]


class ParaWorkerFix:
    def __init__(self, overlap_size=10, adam_num_iter=100):
        self.overlap_size = overlap_size
        self.adam_num_iter = adam_num_iter
        self.op = SplitMergeOp(overlap_size)

    def loss(self, x):
        avg_x = self.op.split(
            self.op.merge(x), x.shape[0], x.shape[-1]
        )
        return th.sum(
            (x - avg_x) ** 2,
            dim=(1, 2, 3),
        )

    def split_noise(self, cur_t, xt):
        noise = simple_noise(cur_t, xt)
        b, _, _, w = xt.shape
        final_img_w = w * b - self.overlap_size * (b - 1)
        noise = rearrange(noise, "(t n) c h w -> t c h (n w)", t=1)[:, :, :, :final_img_w][0]
        noise = self.op.split(noise, b, w)
        return noise

    def adam_grad_weight(self, x0, grad_term, cond_loss_fn):
        init_weight = th.ones(x0.shape[0]).to(x0)
        grad_term = grad_term.detach()
        x0 = x0.detach()
        with th.enable_grad():
            weights = init_weight.requires_grad_()
            optimizer = th.optim.Adam(
                [
                    weights,
                ],
                lr=1e-2,
            )

            def _loss(w):
                cor_x0 = x0 - batch_mul(w, grad_term)
                return cond_loss_fn(cor_x0).sum()

            for _ in range(self.adam_num_iter):
                optimizer.zero_grad()
                _cur_loss = _loss(weights)
                _cur_loss.backward()
                optimizer.step()
        return weights

    def x0_replace(self, x0, sclar_t, thres_t):
        if sclar_t > thres_t:
            merge_x0 = self.op.merge(x0)
            return self.op.split(merge_x0, x0.shape[0], x0.shape[-1])
        else:
            return x0