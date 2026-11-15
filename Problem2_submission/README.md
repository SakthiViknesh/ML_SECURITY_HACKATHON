# Black-Box Adversarial Attack on a CNN (CIFAR-10-style, hard-label only)

This repository contains our solution to **Problem 2: Black Box Adversarial Attack
on CNN**. Given a pretrained CNN that can only be queried for its **predicted
label** (no scores, no gradients, no architecture/parameters), we generate one
adversarial image per clean image that:

- causes the model to predict the **wrong** class, and
- is **visually almost identical** to the original (small, imperceptible noise).

## Results summary

| Metric | Value |
|---|---|
| Attack success rate | **100.0%** (1000 / 1000) |
| Mean L∞ perturbation (0–1 pixel scale) | **0.0188** (≈ 4.8 / 255) |
| Max L∞ perturbation | 0.1126 (≈ 28.7 / 255) |
| Mean L2 perturbation (3072-dim vector) | 0.911 |
| Total queries to the target model | 12,657 |
| Queries per image | **12.7** |

(Full raw numbers are in [`attack_stats.json`](attack_stats.json).)

## Method (short version — full details in `report.pdf`)

The target model only returns a hard label, so there is no gradient or confidence
signal to exploit directly. We use a **transfer-based attack**:

1. Train two independent surrogate CNNs (`ResNet9`, a VGG-style network) from
   scratch on the real, public **CIFAR-10** dataset (same 10 classes / 32×32
   images as this task). See [`surrogate/`](surrogate/).
2. Craft perturbations with standard white-box **PGD** against this surrogate
   ensemble (we *do* have gradients for our own models), relying on the
   well-known **transferability** of adversarial examples across independently
   trained networks solving the same task.
3. Only query the real black-box model to **check** whether a candidate
   perturbation already flips the prediction, trying an increasing L∞ budget
   schedule (4/255 → 32/255) and stopping as soon as one succeeds.
4. **Binary-search** the perturbation's scale (still only using hard-label
   queries) to shrink it down to the minimum that still fools the model —
   this is what keeps the final noise so small.
5. A random-noise fallback exists for any image the transfer attack alone
   cannot flip (none were needed in this run — all 1000 succeeded via
   transfer + PGD).

This keeps the number of queries to the real model very low (~13/image) while
reaching 100% success with imperceptible noise.

## Repository layout

```
p2_code.ipynb          Submission notebook (attack implemented in the designated cells)
p2_data/
  clean_images/        1000 provided clean images
  adv_images/           1000 generated adversarial images (<class>_<idx>_adv.png)
  cnn_model.pt          Provided black-box target model
  readme.txt            Provided data description
surrogate/
  models.py            Surrogate CNN architectures (ResNet9, VGGSmall)
  train.py             Trains a surrogate on real CIFAR-10
  attack.py            Core attack logic (PGD + eps-schedule + binary-search refinement)
  run_full_attack.py   Runs the attack end-to-end on all 1000 images (standalone script)
  resnet9.pt           Trained ResNet9 surrogate weights (82.9% CIFAR-10 test acc)
  vgg.pt               Trained VGG surrogate weights (83.2% CIFAR-10 test acc)
attack_stats.json       Machine-readable summary of the final run
report.pdf              Short written report of the methodology
```

## How to reproduce

### Option A — just run the notebook (uses the already-trained surrogate weights)

1. Place `p2_data.zip` (clean images + `cnn_model.pt`) next to `p2_code.ipynb`,
   matching the zip path set in the notebook's second cell.
2. Open `p2_code.ipynb` and run all cells top to bottom. The notebook will:
   - extract the data,
   - load the target model and the two pretrained surrogates from `surrogate/`,
   - run the attack on all 1000 images,
   - save the adversarial images to `p2_data/adv_images/`.

Requires: `torch`, `torchvision`, `pillow`, `matplotlib` (CPU is fine; no GPU
required, though it will be faster if available).

### Option B — retrain the surrogates yourself first

```bash
cd surrogate
python train.py --arch resnet9 --epochs 3 --out resnet9.pt
python train.py --arch vgg --epochs 3 --out vgg.pt
```

This downloads CIFAR-10 automatically (first run only) and trains both
surrogates (≈35–45 min each on CPU). Then run the notebook as in Option A, or
run the attack standalone:

```bash
cd surrogate
python run_full_attack.py
```

This reads `../p2_data/clean_images/`, writes `../p2_data/adv_images/`, and
writes `../attack_stats.json`.

## Notes on design choices

- **Why a transfer attack instead of a pure decision-based attack** (e.g.
  Boundary Attack / HopSkipJumpAttack)? Those methods work with hard labels
  alone but typically need hundreds to thousands of queries *per image* to
  converge to a small perturbation. Since the clean images are exactly
  CIFAR-10-style data, training our own CIFAR-10 classifiers and exploiting
  transferability gets comparable or better success/imperceptibility at a
  small fraction of the query cost (here, ~13 queries/image on average),
  which also helps on the query-count tie-break criterion.
- **Why two different architectures (ResNet9 + VGG)?** Averaging gradients
  from architecturally distinct networks reduces overfitting to one
  surrogate's specific decision boundary and improves transfer success,
  compared to using a single surrogate.
- **Why binary-search refinement?** Finding *any* successful perturbation is
  easy once eps is large enough; but imperceptibility is scored separately,
  so we spend a small, fixed number of extra queries (10/image) specifically
  to shrink each found perturbation to the minimum magnitude that still
  works, trading a few extra queries for a large imperceptibility gain.
