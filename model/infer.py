import os
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

import torch
from matplotlib import colormaps as cmaps
from config import CFG
from architecture import JointModel


# ===============================
# CONFIG
# ===============================
CKPT_PATH = str(CFG.CKPT_DIR / "fsdam_best.pt")

INPUT_RGB_DIR = "input_rgb"                    # << user folder
OUT_DIR       = "inference_output"             # << results folder

FIXED_Q = CFG.FIXED_Q
MAXTOK = CFG.EVAL_MAX_NEW_TOKENS
BEAMS  = CFG.EVAL_BEAMS


# ===============================
# HELPERS
# ===============================
def apply_jet_overlay(bgr, gaze, alpha=0.45):
    H, W = bgr.shape[:2]
    if gaze.shape != (H, W):
        gaze = cv2.resize(gaze, (W, H))

    g = gaze.astype(np.float32)
    g = g / (g.max() + 1e-8)
    g = g ** 3

    jet = cmaps.get_cmap("jet")
    rgba = jet(g)
    rgb = (rgba[..., :3] * 255).astype(np.uint8)
    bgr_heat = rgb[..., ::-1]

    return (alpha * bgr_heat + (1 - alpha) * bgr).astype(np.uint8)


# ===============================
# MAIN
# ===============================
@torch.no_grad()
def main():

    inp = Path(INPUT_RGB_DIR)
    out_root = Path(OUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    rgb_files = sorted([f for f in inp.glob("*") if f.suffix.lower() in [".jpg", ".png", ".jpeg"]])
    print(f"[infer] found {len(rgb_files)} images")

    # load model
    model = JointModel().to(CFG.DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    ckpt = ckpt.get("state_dict", ckpt)
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    for path in rgb_files:

        # prepare image
        pil = Image.open(path).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

        # gaze forward
        _, fmap = model.vlm.encode([pil])
        pred_gaze = model.gaze_head(fmap).cpu().numpy()[0][0]

        # caption forward
        pred_caption = model.vlm.generate(
            images=[pil],
            prompts=[FIXED_Q],
            max_new_tokens=MAXTOK,
            num_beams=BEAMS
        )[0]

        # create output folder
        name = path.stem
        folder = out_root / name
        folder.mkdir(exist_ok=True)

        # save overlay
        overlay = apply_jet_overlay(bgr, pred_gaze)
        cv2.imwrite(str(folder / "prediction_overlay.png"), overlay)

        # save raw gaze map
        np.save(str(folder / "prediction_gaze.npy"), pred_gaze)

        # save caption json
        with open(folder / "prediction.json", "w") as f:
            json.dump({
                "image": path.name,
                "prediction_caption": pred_caption
            }, f, indent=2)

        print(f"[saved] {name}")

    print("\n[inference done]")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
