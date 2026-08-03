from omegaconf import OmegaConf
import torch

def _clean(name):
    return name.replace("_orig_mod.", "")


def save_checkpoint(cfg, model, ema, epoch, optimizer, out_dir):
    _raw = {_clean(k): v for k,v in model.state_dict().items()} # clean the compiled key
    checkpoint = { 
        "epoch": epoch+1,
        "model": _raw,
        "ema": {
            "params": ema.shadow,
            "steps": ema.steps,
        },
        "optimizer": optimizer.state_dict(),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
    }

    torch.save(checkpoint, out_dir / "checkpoint.pt")