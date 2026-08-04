from vit.evaluation.fid import FID
import torch


def test_simple_calculation():
    fid = FID()

    fid._update_mode(
        mu=torch.ones(fid.feature_size, dtype=torch.float64),
        prod=torch.ones(fid.feature_size, fid.feature_size, dtype=torch.float64),
        total=100,
        mode="real"
    )

    fid._update_mode(
        mu=torch.ones(fid.feature_size, dtype=torch.float64),
        prod=torch.ones(fid.feature_size, fid.feature_size, dtype=torch.float64),
        total=100,
        mode="fake"
    )

    dist = fid.frechet_distance()
    print(dist)