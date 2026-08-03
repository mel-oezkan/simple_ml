# simple_ml

This is a collection of various DL methods for image generation trained on very simple datasets and with small model sizes. 

The goal of this repo is reimplementing existing ideas and understading them better!

## List of possible Improvements
- Plain Diffusion Model: 
- Diffusion Model with 2d embddings
- Modify loss to inlcude cfg
- Improvements from https://github.com/LTH14/JiT/blob/main/model_jit.py:
    - Effectiveness of RMSNorm
    - Delaying the conditioning to later layers
- Spiral Rope https://arxiv.org/pdf/2602.03227
    - Actually use Rope correctly

## Running with Modal

In order to run the training wiht modal we can simply execute the following command.
```Python
uv run modal run modal_train.py 
```
However to utilize the hydra configs we added an additional flag `--overrides`. Using this flag the existing config is overwritten similar to how it happens when usign hydra noramlly

```Python
uv run modal run modal_train.py --overrides "molde.n_blocks=12 batch_size=512"
```

