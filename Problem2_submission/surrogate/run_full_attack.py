"""Run the full transfer + query-refinement black-box attack on all 1000 clean
images and save the adversarial images to p2_data/adv_images/<name>_adv.png.

This script mirrors exactly what is embedded in p2_code.ipynb (cells 8 and 10),
so it can be run standalone for development/testing with the real target model.
"""
import json
import sys
import time

import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, ".")
from models import ResNet9, VGGSmall
from attack import run_attack, normalize, denormalize

DEVICE = "cpu"
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
IMG_PER_CLASS = 100

DATA_DIR = "../p2_data"
ADV_DIR = f"{DATA_DIR}/adv_images"


def main():
    torch.manual_seed(0)

    target = torch.jit.load(f"{DATA_DIR}/cnn_model.pt", map_location=DEVICE)
    target.eval()

    def query_fn(x_norm):
        with torch.no_grad():
            return target(x_norm)

    surrogates = []
    for arch, path in [("resnet9", "resnet9.pt"), ("vgg", "vgg.pt")]:
        m = ResNet9() if arch == "resnet9" else VGGSmall()
        m.load_state_dict(torch.load(path, map_location=DEVICE))
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
        surrogates.append(m)
    print(f"loaded {len(surrogates)} surrogates")

    transform = transforms.ToTensor()
    names = [f"{c}_{i}" for c in CLASS_NAMES for i in range(IMG_PER_CLASS)]
    paths = [f"{DATA_DIR}/clean_images/{n}.png" for n in names]
    labels = [CLASS_NAMES.index(n.rsplit("_", 1)[0]) for n in names]

    X_raw = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths]).to(DEVICE)
    y_true = torch.tensor(labels).to(DEVICE)

    with torch.no_grad():
        clean_preds = query_fn(normalize(X_raw, DEVICE))
    already_wrong = clean_preds != y_true
    print(f"already-misclassified clean images: {already_wrong.sum().item()} / {len(y_true)}")

    attack_mask = ~already_wrong
    idx_attack = attack_mask.nonzero(as_tuple=True)[0]

    t0 = time.time()
    adv_sub, solved_sub, n_queries = run_attack(
        X_raw[idx_attack], y_true[idx_attack], surrogates, query_fn, DEVICE,
        eps_schedule=(4 / 255, 8 / 255, 12 / 255, 16 / 255, 24 / 255, 32 / 255),
        iters_per_eps=25, alpha_frac=0.25,
        refine_steps=10, fallback_eps=64 / 255, fallback_tries=20, verbose=True,
    )
    elapsed = time.time() - t0
    print(f"attack finished in {elapsed:.1f}s, queries={n_queries}")

    adv_raw = X_raw.clone()
    adv_raw[idx_attack] = adv_sub
    solved = already_wrong.clone()
    solved[idx_attack] = solved_sub
    total_queries = n_queries + X_raw.size(0)  # + the initial clean-prediction query batch

    with torch.no_grad():
        final_preds = query_fn(normalize(adv_raw, DEVICE))
    final_success = final_preds != y_true
    assert bool((final_success == solved).all()), "solved mask inconsistent with final re-check"

    linf = (adv_raw - X_raw).abs().view(X_raw.size(0), -1).max(1).values
    l2 = (adv_raw - X_raw).view(X_raw.size(0), -1).norm(dim=1)

    print(f"FINAL success rate: {solved.float().mean().item():.4f} ({solved.sum().item()}/{len(y_true)})")
    print(f"mean L_inf (0-1 scale): {linf.mean().item():.4f}  mean L2: {l2.mean().item():.4f}")
    print(f"total target-model queries: {total_queries}  ({total_queries/len(y_true):.1f} per image)")

    import os
    os.makedirs(ADV_DIR, exist_ok=True)
    for i, n in enumerate(names):
        img = transforms.ToPILImage()(adv_raw[i].clamp(0, 1).cpu())
        img.save(f"{ADV_DIR}/{n}_adv.png")

    stats = {
        "success_rate": solved.float().mean().item(),
        "n_success": int(solved.sum().item()),
        "n_total": len(y_true),
        "n_already_misclassified": int(already_wrong.sum().item()),
        "mean_linf_0_1": linf.mean().item(),
        "max_linf_0_1": linf.max().item(),
        "mean_l2": l2.mean().item(),
        "total_queries": int(total_queries),
        "queries_per_image": total_queries / len(y_true),
        "elapsed_seconds": elapsed,
    }
    with open(f"{DATA_DIR}/../attack_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("saved stats to attack_stats.json")


if __name__ == "__main__":
    main()
