# General Paper Notes


## [EMA Paper](https://arxiv.org/pdf/2411.18704v1)
- Two main overheads (1) determine window for moving average (2) tuning for the correct lr
    - They circumvent the problem of (2) by using cosine annealing and finding best early stopping epoch from validation dataset
- Since the model is realively small we could track multiple models with different window sizes (e.g. M=5)
- Training performance is similar when using EMA but fewer epochs are needed (3/4 of epoch budget would have sufficed)
- With EMA models we can use larger LR du to an assumed consistency (betwen preds during train) coming from the slow moving average of EMA

## Short Background on KID 
Original Paper: [DEMYSTIFYING MMD GANS](https://arxiv.org/pdf/1801.01401)

Advntages: 
- does not assume a parametric form for the distribution (2% of activations don't have a density -> activation is 0)
- KID is an unbiased estimator
- Reduces the amount of data to estimate a correct score

Disadvantage:
- Althoug the estimator is unbiased it still is subject to variance fluctuations [[Ref]](https://arxiv.org/pdf/2103.10428)

## DDIM paper
- having smaller eta is more effective when usign smaller steps sizes
    - when the steps increase eta = 1 seems to recover the origina ddpm performance



## Real world fashion datasets
https://github.com/switchablenorms/deepfashion2

## Paper reading list
- [CLASSIFIER-FREE DIFFUSION GUIDANCE](https://arxiv.org/pdf/2207.12598)
- [Spiral RoPE](https://arxiv.org/pdf/2602.03227)
- [SiT Paper](https://arxiv.org/pdf/2401.08740)

