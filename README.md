# BLASTNet HIT GNN

Train a JAX message-passing graph neural network to predict the next snapshot
of the periodic, compressible, inert CH4-O2 homogeneous-isotropic-turbulence
simulation published by [BLASTNet](https://blastnet.github.io/compressible_inert_ch4o2_hit).
PyVista reads the structured grid and writes prediction/reference fields for
visualization.

The Kaggle dataset contains 99 snapshots (`id000` through `id098`) of eight
fields on a `129 x 129 x 129` grid. The full download is about 6.8 GB. The model
trains on periodic spatial patches, with the earliest 80% of temporal
transitions used for training and the remaining transitions held out for
validation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test]"
```

Download and extract the public Kaggle dataset:

```bash
blastnet-hit-download --data-dir data/raw
```

The downloader fetches files from Kaggle's public dataset API without requiring
an API token. To use the Kaggle CLI instead, configure a Kaggle API token and
run `kaggle datasets download -d waitongchung/inert-ch4o2-hit-dns --unzip -p
data/raw`.

The extracted directory should contain `data/`, `grid/`, and `info.json`.

## Train

```bash
blastnet-hit-train \
  --data-dir data/raw \
  --checkpoint models/hit_gnn.npz \
  --epochs 10
```

The trainer uses three-dimensional periodic message passing over six nearest
neighbors. It predicts all eight flow/species fields at the next snapshot.
Patch-based updates avoid materializing the full graph and its activations.
Training samples, patch size, model width/depth, and learning rate can be
changed with CLI options; run `blastnet-hit-train --help` for details.

## Predict and visualize

```bash
blastnet-hit-visualize \
  --data-dir data/raw \
  --checkpoint models/hit_gnn.npz \
  --step 10 \
  --output artifacts/prediction.vts \
  --screenshot artifacts/density.png \
  --field predicted_density
```

The VTK XML structured-grid file contains predicted and reference values for
all eight fields. The optional screenshot uses an off-screen PyVista plotter.
It renders the middle z-plane to show the interior field.

For an interactive, side-by-side PyVista window of the predicted and reference
oxygen core, launch:

```bash
blastnet-hit-interactive \
  --data-dir data/raw \
  --checkpoint models/hit_gnn.npz
```

Use the slider or Left/Right keys to change snapshots, Space to play or pause,
and drag to rotate the synchronized views. Each prediction is a one-step
forecast conditioned on that frame's DNS input.

## Validation

```bash
python -m pytest
```

## Dataset citation and license

The dataset is distributed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Review the terms before use; they restrict commercial use and require
share-alike distribution.

Please cite the [dataset DOI](https://doi.org/10.5281/zenodo.7242864) and its
associated [Combustion and Flame article](https://doi.org/10.1016/j.combustflame.2021.111758).
