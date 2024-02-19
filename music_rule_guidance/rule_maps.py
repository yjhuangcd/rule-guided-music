from . import music_rules
import torch.nn.functional as F
from functools import partial

FUNC_DICT = {'pitch_hist': music_rules.total_pitch_class_histogram,
             'note_density': music_rules.note_density,
             'note_density_hr_1': partial(music_rules.note_density, horizontal_scale=1.),
             'note_density_hr_2': partial(music_rules.note_density, horizontal_scale=2.),
             'note_density_class': music_rules.note_density_class,
             'chord_progression': music_rules.get_chords,
             # use lower time resolution
             'note_density_pixel': partial(music_rules.note_density, interval=16),
             'chord_progression_pixel': partial(music_rules.get_chords, fs=12.5),
             }


def mse_loss_mean(gen_rule, y_):
    return F.mse_loss(gen_rule.float(), y_.float(), reduction="none").mean(dim=-1)


def zero_one_loss_mean(gen_rule, y_):
    return (y_ != gen_rule).float().mean(dim=-1)


def zero_one_loss_sum(gen_rule, y_):
  return (y_ != gen_rule).float().sum(dim=-1)


# used in beam_sampling to select the best sample, and the loss to report
LOSS_DICT = {'pitch_hist': mse_loss_mean,
             'note_density': mse_loss_mean,
             'note_density_hr_1': mse_loss_mean,
             'note_density_hr_2': mse_loss_mean,
             'note_density_class': zero_one_loss_mean,
             'chord_progression': zero_one_loss_mean,
             'note_density_pixel': mse_loss_mean,
             'chord_progression_pixel': zero_one_loss_mean,
             }
