from vit.evaluation.fid import FID
import torch


def test_simple_calculation():
    fid = FID()

    fid._update_mode(
        mu=torch.ones(fid.feature_size, dtype=torch.float64),
        prod=torch.ones(fid.feature_size, fid.feature_size, dtype=torch.float64),
        total=100,
        mode="real",
    )

    fid._update_mode(
        mu=torch.ones(fid.feature_size, dtype=torch.float64),
        prod=torch.ones(fid.feature_size, fid.feature_size, dtype=torch.float64),
        total=100,
        mode="fake",
    )

    dist = fid.frechet_distance()
    print(dist)


def test_self_distance():
    fid = FID()

    fid_distance = fid.frechet_distance_from_folder(
        folder_real="/content/imagenet-debug",
        folder_fake="/content/imagenet-debug",
    )
    # -5.4788788006021605e-05

    from cleanfid import fid
    cleanfid_score = fid.compute_fid(
        fdir1="/content/imagenet-debug", 
        fdir2="/content/imagenet-debug"
    )
    # -5.943278975450994e-05

    assert torch.allclose(fid_distance, cleanfid_score, atol=(cleanfid_score * 0.05))
