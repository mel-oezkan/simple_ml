from contextlib import contextmanager
import logging

import torch


class EMA:
    def __init__(self, model, decay, current_step=0):
        self.decay = decay

        # will start with the default init params
        self.shadow = {
            # needs to specify named params otherwise alpha, beta will be averaged
            k: v.detach().clone() for k, v in model.named_parameters()
        }

        self.steps = current_step

    @contextmanager
    def averaged(self, model):
        """Context fn to change weights to ema and back."""
        base_backup = {k: v.detach().clone() for k, v in model.state_dict().items()}

        merged = base_backup.copy()
        for k,v in self.shadow.items():
            # check if k exists in merged
            if k not in merged:
                raise KeyError(f"Invalid, key:{k} not found in model state dict")

            merged[k] = v.to(merged[k].dtype)
            
        model.load_state_dict(merged, strict=True)

        try: 
            yield # return the new weights
        finally:
            logging.info("Restoring the original weigts.")
            model.load_state_dict(base_backup)

    @torch.no_grad()
    def update(self, model):
        self.steps += 1

        d_eff = min(self.decay, (1 + self.steps) / (10 + self.steps))
        for k, v in model.named_parameters():
            if v.dtype.is_floating_point:
                (self.shadow[k].mul_(d_eff).add_(v.float(), alpha=1 - d_eff))
