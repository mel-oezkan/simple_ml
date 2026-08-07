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


def test_fid_distance():
    fid = FID()

    fid_distance = fid.frechet_distance_from_folder(
        folder_real="data/imagenet-debug",
        folder_fake="data/imagenet-debug",
    )
    # -5.4788788006021605e-05

    from cleanfid import fid
    cleanfid_score = fid.compute_fid(
        fdir1="data/imagenet-debug", 
        fdir2="data/imagenet-debug"
    )
    # -5.943278975450994e-05

    expected = torch.tensor(
        cleanfid_score,
        dtype=fid_distance.dtype,
        device=fid_distance.device,
    ) 

    assert torch.allclose(fid_distance, expected, rtol=0.05)


def test_kid_distance():
    fid = FID()

    kid_distance = fid.kid_distance_from_folder(
        folder_real="data/imagenet-debug",
        folder_fake="data/imagenet-debug",
    )
    # -0.0164

    from cleanfid import fid
    cleanfid_score = fid.compute_kid(
        fdir1="data/imagenet-debug", 
        fdir2="data/imagenet-debug"
    )
    # -0.01790899597108364
    
    expected = torch.tensor(
        cleanfid_score,
        dtype=kid_distance.dtype,
        device=kid_distance.device,
    ) 

    assert torch.allclose(kid_distance, expected, rtol=0.05)
