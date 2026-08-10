# simple_ml

This is a collection of various DL methods for image generation trained on very simple datasets and with small model sizes. 

The goal of this repo is reimplementing existing ideas and understading them better!


## Running with Modal

In order to run the training wiht modal we can simply execute the following command.
```Python
uv run modal run modal_train.py 
```
However to utilize the hydra configs we added an additional flag `--overrides`. Using this flag the existing config is overwritten similar to how it happens when usign hydra noramlly

```Python
uv run modal run modal_apps/train.py --overrides "molde.n_blocks=12 batch_size=512"
```

## Evaluation   

Download the ImageNet-10K dataset into the repository's ignored `data/` directory:

```bash
mkdir -p data
curl -L -o data/imagenet-10k.zip \
  https://www.kaggle.com/api/v1/datasets/download/priyerana/imagenet-10k
```

Extract it with:

```bash
unzip data/imagenet-10k.zip -d data/imagenet-10k
```
## Testing
For the testing we seperated the training into 3 categories: "gpu", "integration" and "basic". For the first two tests we need to spin up a modal container and run them there. To do so we created `modal_test.py`.
For the _basic_ test you can simply run the following command:
```
uv run pytest -m "not gpu and not integration" 
```

For the modal dependent evaluation you need to run any of the following commands.
```
uv run modal run modal_apps/test.py --suite basic
uv run modal run modal_apps/test.py --suite gpu
uv run modal run modal_apps/test.py --suite integration
uv run modal run modal_apps/test.py --suite all
```

```
uv run python -m scripts.visualize.plot_training \
```


## List of possible Improvements
- Plain Diffusion Model: 
- Diffusion Model with 2d embddings
- Modify loss to inlcude cfg
- Improvements from https://github.com/LTH14/JiT/blob/main/model_jit.py:
    - Effectiveness of RMSNorm
    - Delaying the conditioning to later layers
- Spiral Rope https://arxiv.org/pdf/2602.03227
    - Actually use Rope correctly

## Possible Problems
- When loading the 