from contextlib import contextmanager
import logging

import torch


def _clean(name):
    return name.replace("_orig_mod.", "")

class EMA:
    def __init__(self, model, decay, current_step=0):
        self.decay = decay

        # will start with the default init params
        self.shadow = {
            # needs to specify named params otherwise alpha, beta will be averaged
            _clean(k): v.detach().clone() for k, v in model.named_parameters()
        }

        self.steps = current_step

    @contextmanager
    def averaged(self, model):
        """Context fn to change weights to ema and back."""
        base_backup = {k: v.detach().clone() for k, v in model.state_dict().items()}

        merged = base_backup.copy()
        # we need to iterate over merged and then clean so the match can be found
        for k in merged:
            clean_k = _clean(k)
            if clean_k in self.shadow:
                merged[k] = self.shadow[clean_k].to(merged[k].dtype)
                
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
                (self.shadow[_clean(k)].mul_(d_eff).add_(v.float(), alpha=1 - d_eff))
