
import hydra
from omegaconf import DictConfig

from vit.evaluation.fid import FID

def eval(cfg: DictConfig):
    """Helper function to run the fid eval and store the results"""
    fid = FID()
    if cfg.data.image_dataset:
        fid_distance = fid.frechet_distance_from_folder(
            folder_real=cfg.data.root,
            folder_fake=cfg.data.root,
        )

        print(fid_distance)

    else: 
        raise NotImplementedError("Dataset needs to be a ImageDataset and image_dataset has to be True in the config")

@hydra.main(version_base=None, config_path="../../conf", config_name="eval")
def main(cfg: DictConfig):
    eval(cfg)


if __name__ == "__main__":
    main()