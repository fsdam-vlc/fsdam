import json, random
from pathlib import Path
from typing import List, Dict
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from config import CFG

def _load_json_list(path: Path) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Top-level JSON in {path} must be a list.")
    return data

def _assistant_caption(rec: Dict) -> str:
    if isinstance(rec.get("answer"), str) and rec["answer"].strip():
        return rec["answer"].strip()
    conv = rec.get("conversations", [])
    for t in conv:
        if isinstance(t, dict) and t.get("from") == "gpt":
            return (t.get("value") or "").strip()
    raise ValueError("No caption found.")

class BDDADataset(Dataset):
    def __init__(self, rgb_dir, gaze_dir, json_path, for_train=True):
        self.rgb_dir = Path(rgb_dir)
        self.gaze_dir = Path(gaze_dir)
        self.items = _load_json_list(Path(json_path))
        self.for_train = for_train
        self.records = self._index()
        if for_train:
            random.shuffle(self.records)
        print(f"[dataloader] JSON={Path(json_path).name} | N={len(self.records)}")

    def _index(self):
        out = []
        for rec in self.items:
            name = Path(rec.get("image", "")).name or (rec.get("id", "") + ".png")
            if name:
                out.append({
                    "fname": name,
                    "prompt": CFG.FIXED_Q,
                    "answer": _assistant_caption(rec)
                })
        return out

    def _load_pil(self, fname):
        img = Image.open(self.rgb_dir / fname).convert("RGB")
        return img

    def _load_gaze(self, fname):
        g = Image.open(self.gaze_dir / fname).convert("L").resize((CFG.GAZE_OUT, CFG.GAZE_OUT), Image.BILINEAR)
        arr = np.asarray(g, dtype=np.float32)
        if arr.max() > 1.0:
            arr /= 255.0
        s = float(arr.sum())
        if s <= 1e-8:
            arr[:] = 1.0 / (CFG.GAZE_OUT * CFG.GAZE_OUT)
        else:
            arr /= s
        return torch.from_numpy(arr).unsqueeze(0)

    def __len__(self): return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        return {
            "fname": r["fname"],
            "image": self._load_pil(r["fname"]),
            "gaze": self._load_gaze(r["fname"]),
            "prompt": r["prompt"],
            "answer": r["answer"]
        }

def collate_fn(batch):
    return {
        "fname": [b["fname"] for b in batch],
        "images": [b["image"] for b in batch],
        "gaze": torch.stack([b["gaze"] for b in batch]),
        "prompts": [b["prompt"] for b in batch],
        "answers": [b["answer"] for b in batch],
    }

def make_loaders():
    tr_ds = BDDADataset(CFG.TRAIN_RGB_DIR, CFG.TRAIN_GAZE_DIR, CFG.TRAIN_JSON, True)
    va_ds = BDDADataset(CFG.VAL_RGB_DIR, CFG.VAL_GAZE_DIR, CFG.VAL_JSON, False)
    tr = DataLoader(tr_ds, batch_size=CFG.BATCH_SIZE, shuffle=True,
                    num_workers=CFG.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    va = DataLoader(va_ds, batch_size=CFG.BATCH_SIZE, shuffle=False,
                    num_workers=CFG.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    print(f"[dataloader] Train={len(tr_ds)} | Val={len(va_ds)}")
    return tr, va
