# infer.py
import os
import re
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import DataLoader, Dataset
from matplotlib import colormaps as cmaps

from config import CFG
from architecture import JointModel


# =========================
# Constants from CFG
# =========================
# just plug in path of test set image ; here default path is val set
CKPT_PATH        = str(CFG.CKPT_DIR / "fsdam_best.pt")
IMAGES_DIR       = str(CFG.VAL_RGB_DIR)  
OUT_DIR          = "outputs"
PROMPT           = CFG.FIXED_Q
BATCH_SIZE       = CFG.BATCH_SIZE
MAX_NEW_TOKENS   = CFG.EVAL_MAX_NEW_TOKENS
NUM_BEAMS        = CFG.EVAL_BEAMS
SAVE_JSON_SLOTS  = False
SAVE_RAW_NPY     = False
PREFIX_ON        = CFG.PREFIX_ON


# =========================
# Sentence post processing
# =========================
def _clean_sentence(s: str) -> str:
    s = re.sub(r'^(the\s+image|this\s+image|the\s+scene)\s+(shows|is|depicts)\s*[:,]?\s*', '', s, flags=re.I)
    s = re.sub(r'\b(\w+)(\s+\1)+\b', r'\1', s, flags=re.I)
    s = re.sub(r'\s+', ' ', s).strip()
    if s and s[-1] not in '.!?':
        s += '.'
    return s

def force_four_sentences(text: str) -> str:
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    sents = [s for s in sents if s.strip()]
    sents = [_clean_sentence(s) for s in sents]
    sents = (sents + ["", "", "", ""])[:4]
    sents = [(_clean_sentence(s) if s else ".") for s in sents]
    return " ".join(sents)

def to_four_slots(text: str) -> Dict[str, str]:
    s = force_four_sentences(text)
    parts = re.split(r'(?<=[.!?])\s+', s.strip())
    parts = [p for p in parts if p.strip()]
    parts = (parts + ["", "", "", ""])[:4]
    return {
        "WHAT": parts[0].strip(),
        "WHERE_NOW": parts[1].strip(),
        "WHERE_NEXT": parts[2].strip(),
        "WHY": parts[3].strip(),
    }


# =========================
# Utilities
# =========================
def list_images(root: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    paths: List[Path] = []
    if root.is_file() and root.suffix.lower() in exts:
        paths = [root]
    else:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                paths.append(p)
    return sorted(paths)

def ensure_dirs(base: Path) -> Tuple[Path, Path, Path, Path]:
    gaze_dir    = base / "gaze"
    overlay_dir = base / "overlay"
    caption_dir = base / "caption"
    debug_dir   = base / "debug"
    for d in [gaze_dir, overlay_dir, caption_dir, debug_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return gaze_dir, overlay_dir, caption_dir, debug_dir

def to_uint8(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)

def save_jet_heatmap(prob_1xhxw: np.ndarray, out_png: Path, size_hw: Tuple[int, int]):
    p = prob_1xhxw[0]
    p = p / (p.max() + 1e-8)
    heat = Image.fromarray(to_uint8(p)).resize((size_hw[1], size_hw[0]), Image.BILINEAR)
    heat = np.asarray(heat).astype(np.float32) / 255.0
    jet = cmaps.get_cmap("jet")
    rgba = jet(heat)
    rgb = (rgba[..., :3] * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(rgb).save(out_png)

def save_jet_overlay(img_rgb: Image.Image, prob_1xhxw: np.ndarray, out_png: Path, alpha: float = 0.45):
    H0, W0 = img_rgb.height, img_rgb.width
    p = prob_1xhxw[0]
    p = p / (p.max() + 1e-8)
    heat = Image.fromarray(to_uint8(p)).resize((W0, H0), Image.BILINEAR)
    heat = np.asarray(heat).astype(np.float32) / 255.0
    jet = cmaps.get_cmap("jet")
    rgba = jet(heat)
    jet_rgb = (rgba[..., :3] * 255.0 + 0.5).astype(np.uint8)
    base = np.asarray(img_rgb.convert("RGB"), dtype=np.uint8)
    over = (alpha * jet_rgb + (1.0 - alpha) * base).astype(np.uint8)
    Image.fromarray(over).save(out_png)


# =========================
# VLM text generation
# =========================
@torch.no_grad()
def _prep_batch_for_vlm(vlm, images: List[Image.Image], prompts: List[str]):
    device    = vlm.device
    model     = vlm.model
    processor = vlm.processor
    tok       = processor.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    msgs = []
    for q in prompts:
        q_clean = q.replace("<image>", "").strip()
        msgs.append([{"role": "user",
                      "content": [{"type": "image"}, {"type": "text", "text": q_clean}]}])
    texts = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    batch = processor(images=images, text=texts, return_tensors="pt", padding=True)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            if k == "pixel_values":
                batch[k] = v.to(device, dtype=model.dtype)
            else:
                batch[k] = v.to(device)
    return batch, tok, model, processor

@torch.no_grad()
def generate_plain(vlm, images: List[Image.Image], prompts: List[str],
                   max_new_tokens: int, num_beams: int) -> List[str]:
    batch, tok, model, processor = _prep_batch_for_vlm(vlm, images, prompts)
    gen_out = model.generate(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        pixel_values=batch["pixel_values"],
        image_sizes=batch.get("image_sizes", None),
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=False,
        use_cache=True,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    texts_out = processor.batch_decode(gen_out, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return [t.split("[/INST]")[-1].strip() if "[/INST]" in t else t.strip() for t in texts_out]

@torch.no_grad()
def generate_with_prefix(vlm, images: List[Image.Image], prompts: List[str], prefix: torch.Tensor,
                         max_new_tokens: int, num_beams: int) -> List[str]:
    batch, tok, model, processor = _prep_batch_for_vlm(vlm, images, prompts)
    tok_emb    = model.language_model.get_input_embeddings()
    prompt_ids = batch["input_ids"]
    prompt_emb = tok_emb(prompt_ids)
    B, Lp, D   = prompt_emb.shape

    assert prefix.shape[0] == B and prefix.shape[2] == D
    Lpfx = prefix.size(1)
    inputs_embeds = torch.cat([prefix.to(prompt_emb.dtype).to(vlm.device), prompt_emb], dim=1)

    pad_ids   = torch.full((B, Lpfx), tok.pad_token_id, dtype=prompt_ids.dtype, device=vlm.device)
    input_ids = torch.cat([pad_ids, prompt_ids], dim=1)
    pad_mask  = torch.ones((B, Lpfx), dtype=batch["attention_mask"].dtype, device=vlm.device)
    attention_mask = torch.cat([pad_mask, batch["attention_mask"]], dim=1)

    gen_out = model.generate(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        pixel_values=batch["pixel_values"],
        image_sizes=batch.get("image_sizes", None),
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=False,
        use_cache=True,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )
    texts_out = processor.batch_decode(gen_out, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return [t.split("[/INST]")[-1].strip() if "[/INST]" in t else t.strip() for t in texts_out]


# =========================
# Data wrapper
# =========================
class ImageDirDataset(Dataset):
    def __init__(self, paths: List[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        p = self.paths[idx]
        try:
            img = Image.open(p).convert("RGB")
        except (UnidentifiedImageError, OSError):
            img = Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8))
        return {"path": str(p), "image": img}

def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "paths":  [b["path"] for b in batch],
        "images": [b["image"] for b in batch],
    }


# =========================
# Main
# =========================
@torch.inference_mode()
def main():
    print("[infer] starting")
    device = CFG.DEVICE

    # Seeds
    torch.manual_seed(CFG.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CFG.SEED)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # Model
    model = JointModel().to(device)
    sd_path = Path(CKPT_PATH)
    if not sd_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {sd_path}")
    state = torch.load(sd_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[infer] missing keys: {len(missing)}")
    if unexpected:
        print(f"[infer] unexpected keys: {len(unexpected)}")
    model.eval()
    print(f"[infer] Loaded checkpoint: {sd_path}")

    # IO
    img_root = Path(IMAGES_DIR).resolve()
    out_root = Path(OUT_DIR).resolve()
    gaze_dir, overlay_dir, caption_dir, debug_dir = ensure_dirs(out_root)
    index_rows = []

    paths = list_images(img_root)
    print(f"[infer] Found {len(paths)} images in {img_root}")

    ds = ImageDirDataset(paths)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate)

    try:
        from tqdm import tqdm
        iterator = tqdm(dl, ncols=100)
    except Exception:
        iterator = dl

    # Loop
    for batch in iterator:
        imgs: List[Image.Image] = batch["images"]
        names = [Path(p).stem for p in batch["paths"]]


        valid = [i for i, im in enumerate(imgs) if im.size[0] > 2 and im.size[1] > 2]
        if not valid:
            continue
        imgs = [imgs[i] for i in valid]
        names = [names[i] for i in valid]
        paths_this = [batch["paths"][i] for i in valid]

        # Vision encode
        z_img, fmap = model.vlm.encode(imgs)  # fmap: [B, d_v, Hf, Wf]

        # Gaze prediction
        pred_gaze = model.gaze_head(fmap)              # [B,1,64,64]
        pred_np   = pred_gaze.detach().float().cpu().numpy()

        # Caption branch
        prompts = [PROMPT] * len(imgs)
        if PREFIX_ON:
            q_text = model._contextual_text_emb(imgs, prompts)  # [B, d_lm]
            M_cap  = int(getattr(CFG, "M_GTOK", 16))

            if hasattr(model, "text2multi"):
                q_multi = model.text2multi(q_text)
            elif hasattr(model, "text2multi_cap"):
                q_multi = model.text2multi_cap(q_text)
            else:
                raise AttributeError("JointModel missing text2multi or text2multi_cap")
            q_multi = q_multi.view(q_text.size(0), M_cap, model.vlm.d_lm)  # [B,M,d_lm]

            if hasattr(model, "gcla"):
                gcla_mod = model.gcla
            elif hasattr(model, "gcla_cap"):
                gcla_mod = model.gcla_cap
            elif hasattr(model, "gcla_shared"):
                gcla_mod = model.gcla_shared
            else:
                raise AttributeError("JointModel missing gcla, gcla_cap, or gcla_shared")

            ctx, _  = gcla_mod(q_multi.to(fmap.dtype), fmap)  # [B,M,d_lm], _
            ctx_bar = ctx.mean(dim=1)                         # [B,d_lm]
            prefix  = model.prefix_adapter(ctx_bar)           # [B,Lp,d_lm]

            raw_caps = generate_with_prefix(
                model.vlm, images=imgs, prompts=prompts, prefix=prefix,
                max_new_tokens=MAX_NEW_TOKENS, num_beams=NUM_BEAMS
            )
        else:
            raw_caps = generate_plain(
                model.vlm, images=imgs, prompts=prompts,
                max_new_tokens=MAX_NEW_TOKENS, num_beams=NUM_BEAMS
            )

        caps = [force_four_sentences(t) for t in raw_caps]

        # Save
        for i, name in enumerate(names):
            H, W = imgs[i].height, imgs[i].width
            gaze_png     = gaze_dir / f"{name}.png"
            overlay_png  = overlay_dir / f"{name}.png"
            caption_txt  = caption_dir / f"{name}.txt"
            caption_json = caption_dir / f"{name}.json"
            raw_npy      = gaze_dir / f"{name}.npy"

            save_jet_heatmap(pred_np[i], gaze_png, (H, W))
            save_jet_overlay(imgs[i], pred_np[i], overlay_png, alpha=0.45)
            caption_txt.write_text(caps[i] + "\n", encoding="utf-8")

            if SAVE_JSON_SLOTS:
                slots = to_four_slots(caps[i])
                caption_json.write_text(json.dumps(slots, ensure_ascii=False, indent=2), encoding="utf-8")

            if SAVE_RAW_NPY:
                np.save(raw_npy, pred_np[i])

            index_rows.append({
                "name": name,
                "image": str(paths_this[i]),
                "gaze_png": str(gaze_png),
                "overlay_png": str(overlay_png),
                "caption_txt": str(caption_txt),
            })

    # Sidecar index
    if index_rows:
        tsv_path = out_root / "index.tsv"
        keys = ["name", "image", "gaze_png", "overlay_png", "caption_txt"]
        with tsv_path.open("w", encoding="utf-8") as f:
            f.write("\t".join(keys) + "\n")
            for r in index_rows:
                f.write("\t".join([r[k] for k in keys]) + "\n")

    print(f"[infer] Done. Results in: {out_root}")
    print(f"         gaze:    {gaze_dir}")
    print(f"         overlay: {overlay_dir}")
    print(f"         caption: {caption_dir}")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
