"""
EMA (Exponential Moving Average) weight averaging for MTR.
Reduces weight oscillation during late training, directly lowering minADE/minFDE variance.
"""
import copy
import torch
import torch.nn as nn


class EMAModel:
    """Exponential Moving Average of model parameters.

    Usage:
        ema = EMAModel(model, decay=0.999)
        # After each optimizer.step():
        ema.update(model)
        # For evaluation/inference:
        ema.apply_to(model)  # swaps in EMA weights
        # ... run eval ...
        ema.restore(model)   # restores original weights
    """

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        self.backup_state = None

    def update(self, model):
        with torch.no_grad():
            model_state = model.state_dict()
            for k, v in self.ema_state.items():
                if v.dtype.is_floating_point:
                    v.mul_(self.decay).add_(model_state[k].detach(), alpha=1.0 - self.decay)
                else:
                    v.copy_(model_state[k])

    def apply_to(self, model):
        """Swap model weights with EMA weights. Call restore() to revert."""
        self.backup_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.ema_state, strict=False)

    def restore(self, model):
        """Restore original model weights after apply_to()."""
        if self.backup_state is not None:
            model.load_state_dict(self.backup_state, strict=False)
            self.backup_state = None

    def state_dict(self):
        return {'decay': self.decay, 'ema_state': self.ema_state}

    def load_state_dict(self, state):
        self.decay = state['decay']
        self.ema_state = state['ema_state']
