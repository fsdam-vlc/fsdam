# train.py 
import os
import json
import time
import random
import numpy as np
import torch
from torch.amp import autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import CFG, sanity_check
from dataloader import make_loaders
from architecture import JointModel, make_optimizer


def set_seed(s=CFG.SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


@torch.no_grad()
def evaluate(model: JointModel, loader):
    model.eval()
    logs = []
    for b in loader:
        step = model.forward_eval(b)
        logs.append(step)
    if len(logs) == 0:
        keys = ["loss_total", "loss_gaze", "loss_align", "loss_cap", "kl_sharp", "kl_blur", "gap_pos"]
        return {k: float("nan") for k in keys}
    keys = logs[0].keys()
    return {k: float(np.mean([l[k] for l in logs])) for k in keys}


def _to_float_dict(d):
    out = {}
    for k, v in d.items():
        try:
            out[k] = float(v)
        except Exception:
            out[k] = v
    return out


def log_jsonl(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")


def maybe_ramp_weights(model: JointModel, epoch: int):
    ramp_e = getattr(CFG, "RAMP_EPOCH", 0)
    w_align = CFG.WEIGHTS.get("align_start", 0.0) if epoch <= ramp_e else CFG.WEIGHTS.get("align", 0.0)
    if not CFG.ALIGN_ON:
        w_align = 0.0
    model.set_loss_weights({"align": w_align})


def train():
    print("[boot] train.py starting", flush=True)
    sanity_check()
    set_seed()

    tr_loader, va_loader = make_loaders()
    print(f"[data] N_tr={len(tr_loader.dataset)} N_va={len(va_loader.dataset)}", flush=True)

    model = JointModel().to(CFG.DEVICE)
    optim = make_optimizer(model)
    scheduler = ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=2, threshold=1e-4)
    prev_lrs = [g["lr"] for g in optim.param_groups]

    mp = str(CFG.MIXED_PREC).lower()
    use_cuda = torch.cuda.is_available()
    scaler = None
    if use_cuda and mp in ("fp16", "float16"):
        from torch.cuda.amp import GradScaler as CudaGradScaler
        scaler = CudaGradScaler()
        print("[amp] fp16 autocast + GradScaler", flush=True)
    elif use_cuda and mp in ("bf16", "bfloat16"):
        print("[amp] bf16 autocast", flush=True)
    else:
        print("[amp] full precision", flush=True)

    best = float("inf")
    bad = 0
    PATIENCE = 5
    json_log = str(CFG.CKPT_DIR / "train_log.jsonl")
    print(f"[log] writing to {json_log}", flush=True)

    eff_bs = CFG.BATCH_SIZE * max(1, CFG.GRAD_ACCUM)
    print(f"[train] effective batch size = {eff_bs}", flush=True)

    for epoch in range(1, CFG.EPOCHS + 1):
        t0 = time.time()
        if hasattr(model, "set_epoch"):
            model.set_epoch(epoch)
        maybe_ramp_weights(model, epoch)

        model.train()
        epoch_logs = []
        optim.zero_grad(set_to_none=True)
        last_i = 0

        for i, b in enumerate(tr_loader, 1):
            last_i = i
            if use_cuda and mp in ("bf16", "bfloat16"):
                with autocast("cuda", dtype=torch.bfloat16):
                    loss, step = model.forward_train(b)
            elif use_cuda and mp in ("fp16", "float16"):
                with autocast("cuda", dtype=torch.float16):
                    loss, step = model.forward_train(b)
            else:
                loss, step = model.forward_train(b)

            if not torch.isfinite(loss):
                print(f"[warn] non finite loss at step {i}, skipping", flush=True)
                optim.zero_grad(set_to_none=True)
                continue

            if scaler is not None and mp in ("fp16", "float16"):
                loss = loss / CFG.GRAD_ACCUM
                scaler.scale(loss).backward()
            else:
                (loss / CFG.GRAD_ACCUM).backward()

            if i % CFG.GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)
                if scaler is not None and mp in ("fp16", "float16"):
                    scaler.step(optim)
                    scaler.update()
                else:
                    optim.step()
                optim.zero_grad(set_to_none=True)
                if hasattr(model, "update_ema"):
                    model.update_ema()

            epoch_logs.append(step)

        if last_i % CFG.GRAD_ACCUM != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.GRAD_CLIP)
            if scaler is not None and mp in ("fp16", "float16"):
                scaler.step(optim)
                scaler.update()
            else:
                optim.step()
            optim.zero_grad(set_to_none=True)
            if hasattr(model, "update_ema"):
                model.update_ema()

        tr = {k: float(np.mean([l[k] for l in epoch_logs])) for k in epoch_logs[0].keys()}
        va = evaluate(model, va_loader)

        scheduler.step(va["loss_total"])
        new_lrs = [g["lr"] for g in optim.param_groups]
        if new_lrs != prev_lrs:
            print(f"[lr] reduced -> {new_lrs}", flush=True)
            prev_lrs = new_lrs

        dt = time.time() - t0
        print(
            f"[Epoch {epoch:02d}] "
            f"Train total={tr['loss_total']:.4f} gaze={tr['loss_gaze']:.4f} "
            f"align={tr['loss_align']:.4f} cap={tr['loss_cap']:.4f} | "
            f"Val total={va['loss_total']:.4f} gaze={va['loss_gaze']:.4f} "
            f"align={va['loss_align']:.4f} cap={va['loss_cap']:.4f} | "
            f"kl_s={va['kl_sharp']:.4f} kl_b={va['kl_blur']:.4f} gap={va['gap_pos']:.4f} | "
            f"{dt:.1f}s",
            flush=True,
        )

        torch.save(model.state_dict(), CFG.CKPT_DIR / "fsdam_last.pt")
        if va["loss_total"] < best - 1e-4:
            best = va["loss_total"]
            torch.save(model.state_dict(), CFG.CKPT_DIR / "fsdam_best.pt")
            print("  ✓ saved best", flush=True)
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"Early stopping at epoch {epoch}", flush=True)

        payload = {
            "epoch": epoch,
            "seconds": float(dt),
            "train": _to_float_dict({
                "loss_total": tr["loss_total"],
                "loss_gaze": tr["loss_gaze"],
                "loss_align": tr["loss_align"],
                "loss_cap":   tr["loss_cap"],
                "kl_sharp":   tr.get("kl_sharp", float("nan")),
                "kl_blur":    tr.get("kl_blur", float("nan")),
                "gap_pos":    tr.get("gap_pos", float("nan")),
            }),
            "val": _to_float_dict(va),
            "best_val": float(best),
            "lr_groups": new_lrs,
            "early_stop": bad >= PATIENCE,
            "eff_batch": eff_bs,
        }
        log_jsonl(json_log, payload)

        if bad >= PATIENCE:
            break

    print("[done] training finished", flush=True)


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        import traceback
        print("[error] Unhandled exception:", repr(e), flush=True)
        traceback.print_exc()
