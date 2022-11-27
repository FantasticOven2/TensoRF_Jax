"""Training script for mic dataset.

For helptext, try running:
```
python train_mic.py --help
```
"""

import pathlib

import fifteen
import tyro

import tensorf.train_config
import tensorf.training

if __name__ == "__main__":
    # Open PDB after runtime errors.
    fifteen.utils.pdb_safety_net()
    # Default configuration for lego dataset.
    lego_config = tensorf.train_config.TensorfConfig(
        run_dir=pathlib.Path(f"./runs/mic-{fifteen.utils.timestamp()}"),
        dataset_path=pathlib.Path("./data/nerf_synthetic/mic"),
        dataset_type="blender",
        n_iters=30000,
        initial_aabb_min=(-1.2638, -0.9567, -0.7677),
        initial_aabb_max=(0.7913,  1.0984,  1.1693),
        appearance_feat_dim=48,
        density_feat_dim=16,
        feature_n_freqs=2,
        viewdir_n_freqs=2,
        grid_dim_init=128,
        grid_dim_final=300,
        upsamp_iters=(2000, 3000, 4000, 5500, 7000),
        # upsamp_iters=(2000, 5500, 7000)
    )

    # Parse arguments.
    config = tyro.cli(
        tensorf.train_config.TensorfConfig,
        default=lego_config,
    )

    # Run training loop!
    tensorf.training.run_training_loop(config)