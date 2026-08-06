"""SwanLab logger — drop-in replacement for tensorboardX.SummaryWriter.

Wraps swanlab.init / swanlab.log / swanlab.finish so existing MTR code
calling tb_log.add_scalar(...) works unchanged.

SwanLab groups metrics by the first segment before '/'.
Tags are organized into three groups:
  - model/*   : loss, learning_rate, per-layer losses
  - train/*   : total_norm, ADE per type
  - eval/*    : 3s/5s/8s evaluation metrics

Usage in train.py / test.py:
    from mtr.utils.swanlab_logger import SwanLabWriter
    tb_log = SwanLabWriter(project='MTR', name=extra_tag, log_dir=...)
"""
import os

import math

class SwanLabWriter:
    # Group prefix mapping: tag prefix -> SwanLab group name
    GROUP_PREFIXES = ['model', 'train', 'eval']

    def __init__(self, project='MTR', name=None, log_dir=None, config=None, **kwargs):
        import swanlab
        self._swanlab = swanlab
        self._global_step = 0
        self._initialized = False
        self._pending = {}  # group -> {tag: value}
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

    def _get_group(self, tag):
        """Extract group name from tag (first segment before '.')."""
        for prefix in self.GROUP_PREFIXES:
            if tag.startswith(prefix + '.'):
                return prefix
        return 'default'

    def add_scalar(self, tag, value, step=None):
        tag = tag.replace('/', '.')

        # Skip non-metric keys from Waymo eval
        if not self._is_valid_metric_tag(tag):
            return

        self._global_step += 1
        # Use '/' as separator — SwanLab groups by first '/' segment
        swanlab_tag = tag.replace('.', '/', 1) if '.' in tag else tag

        # Log-transform loss metrics for better visualization scale
        val = float(value)
        if 'loss' in tag.lower():
            val = math.log(max(val, 1e-8))
            swanlab_tag = swanlab_tag + '/log'

        self._swanlab.log({swanlab_tag: val}, step=self._global_step)

    def add_text(self, tag, text, step=None):
        tag = tag.replace('/', '.')
        self._global_step += 1
        swanlab_tag = tag.replace('.', '/', 1) if '.' in tag else tag
        self._swanlab.log({swanlab_tag: self._swanlab.Text(text)}, step=self._global_step)

    def flush(self):
        pass

    def close(self):
        if self._initialized:
            self._swanlab.finish()
            self._initialized = False

    @staticmethod
    def _is_valid_metric_tag(tag):
        """Filter out Waymo eval keys that are not useful as SwanLab metrics."""
        if '---' in tag or 'Note that' in tag:
            return False
        if '_' in tag and any(t in tag for t in ['TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_CYCLIST']):
            return False
        parts = tag.split()
        if len(parts) >= 2 and parts[-1].isdigit():
            return False
        return True
