import torch as th
from .generic_sampler import batch_mul

def get_x0_grad_pred_fn(raw_net_model, cond_loss_fn, weight_fn, x0_update, thres_t):
    def fn(xt, scalar_t):
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

        if x0_update:
            x0 = x0_update(x0_cor, scalar_t)
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
