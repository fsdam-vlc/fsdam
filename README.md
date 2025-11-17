# FSDAM: Few-Shot Driver Attention Modeling visa Vision-Language Coupling
<video width="720" controls>
  <source src="assets/prediction_vid_01.mp4" type="video/mp4">
</video>


## 1. Setup

```bash
conda create -n fsdam python=3.10 -y
conda activate fsdam
pip install -r model/requirements.txt
````

## 2. Repository layout

```
benchmarking_results/
  bdda_test.json
  dada_2000_test.json
  dreyeve_test.json

eval/
  bleu.py
  meteor.py
  rouge.py
  cider.py
  ciderR.py
  saliency_metrices.py   # adopted from LLada

model/
  dataset/
    bdda/{camera, gaze}
    dada2000/{camera, gaze}
    dreyeve/{camera, gaze}
  outputs/               # default output root
  architecture.py
  config.py              # all paths and hyperparameters
  dataloader.py
  infer.py               # inference, saves gaze maps, overlays, captions
  train.py               # training entrypoint that reads config.py
```

### Data expectation

* Put RGB frames under `model/dataset/<name>/camera`.
* Put GT gaze maps under `model/dataset/<name>/gaze`.
* Filenames must match by stem, for example `000123.jpg` with `000123.png`.

## 3. Inference

`model/infer.py` takes an image folder and an output folder. It loads the checkpoint and paths from `model/config.py`.

```bash
python model/infer.py \
  --images model/dataset/bdda/camera \
  --out model/outputs/bdda_pred
```

You can swap the image folder to run on other sets:

```bash
# DR(eye)VE
python model/infer.py --images model/dataset/dreyeve/camera --out model/outputs/dreyeve_pred

# DADA-2000
python model/infer.py --images model/dataset/dada2000/camera --out model/outputs/dada_pred
```

Outputs:

```
model/outputs/<set>_pred/
  gaze/XXXXXX.npy        # S x S float32, nonnegative
  overlay/XXXXXX.png     # fused map
  caption/XXXXXX.txt
```

> If you want to override the checkpoint path, set it in `model/config.py`.

## 4. Training

Edit `model/config.py` to set dataset roots, batch size, epochs, optimizer, and checkpoint paths. Then run:

```bash
python model/train.py
```

This builds loaders from `model/dataloader.py`, model from `model/architecture.py`, and saves checkpoints to the path defined in `model/config.py`.

## 5. Metrics

`eval/saliency_metrices.py` implements CC, KL, and SIM adopted from LLada. Import in your own scripts if needed:

```python
from eval.saliency_metrices import CC, KLDivergence, SIM
```

Inputs must be tensors shaped `[B, 1, H, W]` with matching sizes. 



## 7. License and citation

Please cite FSDAM if you use this code. Also cite datasets and LLada metrics where used.


