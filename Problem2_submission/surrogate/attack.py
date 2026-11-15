"""Black-box transfer attack: craft adversarial examples using gradients from a
surrogate ensemble (trained independently of the target CNN) and only query the
target model's predicted label to (a) decide when an attack has succeeded and
(b) binary-search the minimal perturbation magnitude along the found direction.

This file is a standalone/testable version of the logic embedded in p2_code.ipynb.
"""
import torch
import torch.nn.functional as F

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def normalize(x_raw, device):
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    return (x_raw - mean) / std


def denormalize(x_norm, device):
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    return x_norm * std + mean


def ensemble_grad(x_raw, y, surrogates, device):
    """Gradient of mean cross-entropy loss (w.r.t. raw [0,1] pixels) across the ensemble."""
    x_raw = x_raw.clone().detach().requires_grad_(True)
    total_loss = 0.0
    for m in surrogates:
        logits = m(normalize(x_raw, device))
        total_loss = total_loss + F.cross_entropy(logits, y)
    total_loss = total_loss / len(surrogates)
    grad = torch.autograd.grad(total_loss, x_raw)[0]
    return grad.detach()


def pgd_attack(x_clean_raw, y, surrogates, device, eps, alpha, iters, x_init_raw=None):
    """Untargeted L_inf PGD in raw [0,1] pixel space against the surrogate ensemble."""
    if x_init_raw is None:
        x_adv = x_clean_raw.clone()
    else:
        x_adv = x_init_raw.clone()
        delta = torch.clamp(x_adv - x_clean_raw, -eps, eps)
        x_adv = torch.clamp(x_clean_raw + delta, 0.0, 1.0)

    for _ in range(iters):
        grad = ensemble_grad(x_adv, y, surrogates, device)
        x_adv = x_adv + alpha * grad.sign()
        delta = torch.clamp(x_adv - x_clean_raw, -eps, eps)
        x_adv = torch.clamp(x_clean_raw + delta, 0.0, 1.0)
    return x_adv.detach()


@torch.no_grad()
def query_labels(query_fn, x_raw_batch, device, batch_size=256):
    """query_fn takes a normalized tensor batch and returns predicted label indices."""
    preds = []
    for i in range(0, x_raw_batch.size(0), batch_size):
        chunk = x_raw_batch[i:i + batch_size]
        preds.append(query_fn(normalize(chunk, device)))
    return torch.cat(preds, dim=0)


def run_attack(X_raw, y_true, surrogates, query_fn, device,
                eps_schedule=(4 / 255, 8 / 255, 12 / 255, 16 / 255, 24 / 255, 32 / 255),
                iters_per_eps=25, alpha_frac=0.25,
                refine_steps=10, fallback_eps=64 / 255, fallback_tries=20, verbose=True):
    """Returns adv_raw (N,3,32,32) in [0,1], success mask, and total query count."""
    n = X_raw.size(0)
    adv_raw = X_raw.clone()
    solved = torch.zeros(n, dtype=torch.bool, device=device)
    best_direction = torch.zeros_like(X_raw)  # adv - clean, direction used for refinement
    n_queries = 0

    x_running = X_raw.clone()  # warm-start across eps levels

    for eps in eps_schedule:
        unsolved_idx = (~solved).nonzero(as_tuple=True)[0]
        if unsolved_idx.numel() == 0:
            break
        alpha = alpha_frac * eps
        x_sub = pgd_attack(X_raw[unsolved_idx], y_true[unsolved_idx], surrogates, device,
                            eps=eps, alpha=alpha, iters=iters_per_eps,
                            x_init_raw=x_running[unsolved_idx])
        x_running[unsolved_idx] = x_sub

        preds = query_labels(query_fn, x_sub, device)
        n_queries += x_sub.size(0)
        success = preds != y_true[unsolved_idx]
        success_global = unsolved_idx[success]
        adv_raw[success_global] = x_sub[success]
        best_direction[success_global] = x_sub[success] - X_raw[success_global]
        solved[success_global] = True
        if verbose:
            print(f"eps={eps:.4f}  newly solved={success.sum().item()}  "
                  f"total solved={solved.sum().item()}/{n}")

    # fallback: random-noise search for any still unsolved images
    unsolved_idx = (~solved).nonzero(as_tuple=True)[0]
    if unsolved_idx.numel() > 0 and verbose:
        print(f"fallback random search for {unsolved_idx.numel()} images")
    for idx in unsolved_idx.tolist():
        x0 = X_raw[idx:idx + 1]
        y0 = y_true[idx:idx + 1]
        found = False
        for _ in range(fallback_tries):
            noise = (torch.rand_like(x0) * 2 - 1) * fallback_eps
            cand = torch.clamp(x0 + noise, 0.0, 1.0)
            pred = query_labels(query_fn, cand, device)
            n_queries += 1
            if pred.item() != y0.item():
                adv_raw[idx] = cand[0]
                best_direction[idx] = cand[0] - x0[0]
                solved[idx] = True
                found = True
                break
        if not found:
            # last resort: large uniform noise almost always flips a CNN's prediction
            noise = (torch.rand_like(x0) * 2 - 1) * (2 * fallback_eps)
            cand = torch.clamp(x0 + noise, 0.0, 1.0)
            pred = query_labels(query_fn, cand, device)
            n_queries += 1
            adv_raw[idx] = cand[0]
            best_direction[idx] = cand[0] - x0[0]
            solved[idx] = (pred.item() != y0.item())

    # binary-search refinement: shrink perturbation magnitude along the found direction
    solved_idx = solved.nonzero(as_tuple=True)[0]
    if solved_idx.numel() > 0:
        lo = torch.zeros(solved_idx.numel(), device=device)
        hi = torch.ones(solved_idx.numel(), device=device)
        x0 = X_raw[solved_idx]
        d = best_direction[solved_idx]
        for _ in range(refine_steps):
            mid = (lo + hi) / 2
            cand = torch.clamp(x0 + mid.view(-1, 1, 1, 1) * d, 0.0, 1.0)
            pred = query_labels(query_fn, cand, device)
            n_queries += cand.size(0)
            is_adv = pred != y_true[solved_idx]
            hi = torch.where(is_adv, mid, hi)
            lo = torch.where(is_adv, lo, mid)
        final = torch.clamp(x0 + hi.view(-1, 1, 1, 1) * d, 0.0, 1.0)
        adv_raw[solved_idx] = final

    return adv_raw, solved, n_queries
