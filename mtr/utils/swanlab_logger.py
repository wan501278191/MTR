"""SwanLab logger — drop-in replacement for tensorboardX.SummaryWriter.

Wraps swanlab.init / swanlab.log / swanlab.finish so existing MTR code
calling tb_log.add_scalar(...) works unchanged.

Usage in train.py / test.py:
    from mtr.utils.swanlab_logger import SwanLabWriter
    tb_log = SwanLabWriter(project='MTR', name=extra_tag, log_dir=...)
"""
import os


class SwanLabWriter:
    def __init__(self, project='MTR', name=None, log_dir=None, config=None, **kwargs):
        import swanlab
        self._swanlab = swanlab
        self._global_step = 0
        self._initialized = False
        self._init_kwargs = dict(
            project=project,
            name=name or 'train',
            log_dir=log_dir,
            description='MTR trajectory prediction training',
        )
        if config:
            self._init_kwargs['config'] = config
        self._init_kwargs.update(kwargs)
        self.init()

    def init(self):
        if not self._initialized:
            self._swanlab.init(**self._init_kwargs)
            self._initialized = True

    def add_scalar(self, tag, value, step=None):
        tag = tag.replace('/', '.')

        # Skip non-metric keys from Waymo eval (separators, breakdown names, etc.)
        if not self._is_valid_metric_tag(tag):
            return

        # SwanLab requires step to be monotonically increasing.
        # Use a global counter instead of the caller-provided step
        # to avoid conflicts between train (accumulated_iter) and eval (epoch) steps.
        self._global_step += 1
        self._swanlab.log({tag: float(value)}, step=self._global_step)

    def add_text(self, tag, text, step=None):
        tag = tag.replace('/', '.')
        self._global_step += 1
        self._swanlab.log({tag: self._swanlab.Text(text)}, step=self._global_step)

    def flush(self):
        pass

    def close(self):
        if self._initialized:
            self._swanlab.finish()
            self._initialized = False

    @staticmethod
    def _is_valid_metric_tag(tag):
        """Filter out Waymo eval keys that are not useful as SwanLab metrics."""
        # Skip separator lines
        if '---' in tag or 'Note that' in tag:
            return False
        # Skip breakdown-level keys like 'minFDE - TYPE_CYCLIST_15'
        # (keep only aggregated keys like 'minFDE - VEHICLE', 'minFDE - PEDESTRIAN', 'minFDE - CYCLIST', 'mAP', etc.)
        if '_' in tag and any(t in tag for t in ['TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST']):
            return False
        # Skip keys with numeric suffixes (breakdown names)
        parts = tag.split()
        if len(parts) >= 2 and parts[-1].isdigit():
            return False
        return True
