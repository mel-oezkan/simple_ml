import logging

from omegaconf import DictConfig, OmegaConf
import torch

from vit.diffusion import Diffusion


def _clean(name):
    return name.replace("_orig_mod.", "")


def save_checkpoint(cfg, model, ema, epoch, optimizer, out_dir):
    _raw = {
        _clean(k): v for k, v in model.state_dict().items()
    }  # clean the compiled key
    checkpoint = {
        "epoch": epoch + 1,
        "model": _raw,
        "ema": {
            "params": ema.shadow,
            "steps": ema.steps,
        },
        "optimizer": optimizer.state_dict(),
        "cfg": OmegaConf.to_container(cfg, resolve=True),
    }

    torch.save(checkpoint, out_dir / "checkpoint.pt")


def prepare_model(cfg: DictConfig, device):
    # Import lazily because EMA uses _clean from this module.
    from vit.ema import EMA

    # check if from_checkpoint is active
    if cfg.training.get("from_checkpoint", False):
        logging.warning("Loading model state from checkpoint")
        return load_checkpoint(cfg, device)

    model = Diffusion(**cfg["model"]).to(device)
    if cfg.training.get("compile", False):
        model.diff_model = torch.compile(model.diff_model)

    ema = EMA(model, cfg.ema.decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)

    return model, ema, optimizer


def load_checkpoint(cfg: DictConfig, device: str):
    from vit.ema import EMA

    checkpoint = torch.load(
        cfg.training.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    checkpoint_cfg = OmegaConf.create(checkpoint["cfg"])

    model = Diffusion(**checkpoint_cfg["model"])
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)

    if checkpoint_cfg.training.get("compile", False):
        model.diff_model = torch.compile(model.diff_model)

    ema = EMA(model, checkpoint_cfg.ema.decay, current_step=checkpoint["ema"]["steps"])
    ema.shadow = {
        name: param.to(device)
        for name, param in checkpoint["ema"]["params"].items()
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=checkpoint_cfg.learning_rate)
    optimizer.load_state_dict(checkpoint["optimizer"])

    return model, ema, optimizer
