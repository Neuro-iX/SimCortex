# src/simcortex/initial_surf/cli.py

import hydra
from omegaconf import OmegaConf
from simcortex.initial_surf.generate import run_generation

@hydra.main(
    version_base="1.1",
    config_path="../../../configs/initial_surf",
    config_name="generate"
)
def generate_app(cfg):
    print(OmegaConf.to_yaml(cfg))
    run_generation(cfg)

if __name__ == "__main__":
    generate_app()
