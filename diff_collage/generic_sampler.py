from collections import defaultdict

import math
import numpy as np
import torch as th
from tqdm import tqdm

__all__ = [
    "generic_sampler",
    "SimpleWork",
]


def batch_mul(a, b):  # pylint: disable=invalid-name
    return th.einsum("a...,a...->a...", a, b)

class SimpleWork:
    def __init__(self, shape, eps_scalar_t_fn):
        self.shape = shape
        self.eps_scalar_t_fn = eps_scalar_t_fn

    def generate_xT(self, n):
        return 80.0 * th.randn((n , *self.shape)).cuda()

    def x0_fn(self, xt, scalar_t, y=None):
        cur_eps = self.eps_scalar_t_fn(xt, scalar_t, y=y)
        x0 = xt - scalar_t * cur_eps
        x0 = th.clip(x0, -1,1)
        return x0, {}, {"x0": x0.cpu()}

    def noise(self, xt, scalar_t):
        del scalar_t
        return th.randn_like(xt)

    def  rev_ts(self, n_step, ts_order):
        _rev_ts = th.pow(
            th.linspace(
                np.power(80.0, 1.0 / ts_order),
                np.power(1e-3, 1.0 / ts_order),
                n_step + 1
            ),
            ts_order
        )
        return _rev_ts.cuda()

def generic_sampler(  # pylint: disable=too-many-locals
    x,
    rev_ts,
    noise_fn,
    x0_pred_fn,
    xt_lgv_fn=None,
    s_churn = 0.0,
    before_step_fn=None,
    end_fn=None, # to do???
    is_tqdm=True,
    is_traj=True,
):
    measure_loss = defaultdict(list)
    traj = defaultdict(list)
    if callable(x):
        x = x()
    if traj:
        traj["xt"].append(x.cpu())

    s_t_min = 0.05
    s_t_max = 50.0
    s_noise = 1.003
    eta = min(s_churn / len(rev_ts), math.sqrt(2.0) - 1)

    loop = zip(rev_ts[:-1], rev_ts[1:])
    if is_tqdm:
        loop = tqdm(loop)

    running_x = x
    for cur_t, next_t in loop:
        # cur_x = traj["xt"][-1].clone().to("cuda")
        cur_x = running_x
        if cur_t < s_t_max and cur_t > s_t_min:
            hat_cur_t = cur_t + eta * cur_t
            cur_noise = noise_fn(cur_x, cur_t)
            cur_x = cur_x + s_noise * cur_noise * th.sqrt(hat_cur_t ** 2 - cur_t ** 2)
            cur_t = hat_cur_t

        if before_step_fn is not None:
            # TODO: may change the callabck
            cur_x = before_step_fn(cur_x, cur_t)

        x0, loss_info, traj_info = x0_pred_fn(cur_x, cur_t)
        epsilon_1 = (cur_x - x0) / cur_t

        xt_next = x0 + next_t * epsilon_1

        x0, loss_info, traj_info = x0_pred_fn(xt_next, next_t)
        epsilon_2 = (xt_next - x0) / next_t

        xt_next = cur_x + (next_t - cur_t) * (epsilon_1 + epsilon_2) / 2

        running_x = xt_next

        if is_traj:
            for key, value in loss_info.items():
                measure_loss[key].append(value)

            for key, value in traj_info.items():
                traj[key].append(value)
            traj["xt"].append(running_x.to("cpu").detach())

        if xt_lgv_fn:
            raise RuntimeError("Not implemented")

    if is_traj:
        return traj, measure_loss
    return running_x
