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
        self._step_offset = {}
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
        if step is None:
            step = self._step_offset.get(tag, 0)
            self._step_offset[tag] = step + 1
        self._swanlab.log({tag: float(value)}, step=step)

    def add_text(self, tag, text, step=None):
        tag = tag.replace('/', '.')
        if step is None:
            step = self._step_offset.get(tag, 0)
            self._step_offset[tag] = step + 1
        self._swanlab.log({tag: self._swanlab.Text(text)}, step=step)

    def flush(self):
        pass

    def close(self):
        if self._initialized:
            self._swanlab.finish()
            self._initialized = False
