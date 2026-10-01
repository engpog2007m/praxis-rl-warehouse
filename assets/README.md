# Assets

The experiments read only these small derived files, which are committed:

| File | Content | Produced by |
|---|---|---|
| `ycb/ycb_task_registry.json` | payload masses for the five task profiles | `scripts/build_ycb_registry.py` |
| `sku110k/density_params.json` | sensor-noise sigmas | `scripts/extract_sku110k_density.py` |
| `ycb/gazebo_models/` | converted Gazebo models | `scripts/build_ycb_registry.py` and related |
| `mutschler/` | warehouse map geometry | `scripts/extract_mutschler_map.py` |

The source repositories are not committed (about 200 MB). They are needed only to
regenerate the files above. To restore the exact versions used:

```bash
git clone https://github.com/elpis-lab/YCB_Dataset.git assets/ycb/models_repo
git -C assets/ycb/models_repo checkout 9e8c6488a2ff673d9aa48a91492fb89423c1b106

git clone https://github.com/eg4000/SKU110K_CVPR19.git assets/sku110k/repo
git -C assets/sku110k/repo checkout 1fb83d6d0e5fd0fef8d53d25940ebd6f8d027020
```
