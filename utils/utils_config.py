import os
import yaml
from easydict import EasyDict as edict
from utils.console_logger import ConsoleLogger
from torch.utils.tensorboard import SummaryWriter


# Top-level YAML key that names a parent config to inherit from. The parent
# is resolved relative to the child file's directory (or absolute). Values
# in the child override the parent; nested dicts are deep-merged. Lists are
# replaced wholesale (so e.g. lr_decay: [200] in the child fully replaces
# lr_decay: [300] in the parent — no element merging).
_INHERIT_KEY = "_inherit"


def _deep_merge(base, override):
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for k, v in override.items():
            merged[k] = _deep_merge(merged[k], v) if k in merged else v
        return merged
    return override


def _load_yaml_with_inherit(path):
    path = os.path.abspath(path)
    with open(path) as fin:
        cfg = yaml.safe_load(fin) or {}
    parent_ref = cfg.pop(_INHERIT_KEY, None)
    if parent_ref is None:
        return cfg
    parent_path = parent_ref if os.path.isabs(parent_ref) else os.path.join(
        os.path.dirname(path), parent_ref
    )
    parent_cfg = _load_yaml_with_inherit(parent_path)
    return _deep_merge(parent_cfg, cfg)


def load_config(path):
    return edict(_load_yaml_with_inherit(path))

def add_log(configs, log_dir=None):
    LOG = ConsoleLogger(configs.task, 'train', log_dir=log_dir)
    logdir = LOG.getLogFolder()
    LOG.info(configs)
    train_summary_writer = SummaryWriter(logdir, 'train')
    return logdir, LOG, train_summary_writer

class LearningRateLambda():
    def __init__(self, decay_schedule, *,
                 decay_factor=0.1,
                 decay_epochs=1.0,
                 warm_up_start_epoch=0,
                 warm_up_epochs=2.0,
                 warm_up_factor=0.01,
                 warm_restart_schedule=None,
                 warm_restart_duration=0.5):
        self.decay_schedule = decay_schedule
        self.decay_factor = decay_factor
        self.decay_epochs = decay_epochs
        self.warm_up_start_epoch = warm_up_start_epoch
        self.warm_up_epochs = warm_up_epochs
        self.warm_up_factor = warm_up_factor
        self.warm_restart_schedule = warm_restart_schedule
        self.warm_restart_duration = warm_restart_duration

    def __call__(self, step_i):
        lambda_ = 1.0

        if step_i <= self.warm_up_start_epoch:
            lambda_ *= self.warm_up_factor
        elif self.warm_up_start_epoch < step_i < self.warm_up_start_epoch + self.warm_up_epochs:
            lambda_ *= self.warm_up_factor**(
                1.0 - (step_i - self.warm_up_start_epoch) / self.warm_up_epochs
            )

        for d in self.decay_schedule:
            if step_i >= d + self.decay_epochs:
                lambda_ *= self.decay_factor
            elif step_i > d:
                lambda_ *= self.decay_factor**(
                    (step_i - d) / self.decay_epochs
                )

        for r in self.warm_restart_schedule:
            if r <= step_i < r + self.warm_restart_duration:
                lambda_ = lambda_**(
                    (step_i - r) / self.warm_restart_duration
                )

        return lambda_