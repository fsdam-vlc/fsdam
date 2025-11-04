#!/usr/bin/env python3
import os, time, json, re, base64, csv
from typing import List, Dict, Tuple, Optional, DefaultDict
from collections import defaultdict
from tqdm import tqdm
import cv2
from openai import OpenAI

# ================== CONFIG ==================
MODEL_ID       = "gpt-4o"   
# rgb & corresponding fused map (overlay)
RGB_DIR        = "dataset/BDDA/test/preprocessed/camera"  # <vid>_<idx>.png
OVERLAY_DIR    = "dataset/BDDA/test/preprocessed/overlay"  # <vid>_<idx>.png

OUT_JSON       = "captions_gpt_test_clean.json"
OUT_CSV        = "captions_gpt_test_clean.csv"
RESUME_JSON    = "captions_gpt_test_v3.progress.json"  

LIMIT          = None       # e.g., 500 or None 
RPM_TARGET     = 2          #  (requests/min)
DELTA_MIN      = 1
DELTA_MAX      = 18
JPEG_WIDTH     = 768        
JPEG_QUALITY   = 85
MAX_WORDS      = 25         
PROMPT_VERSION = "v2.0"
# ============================================

# --------- HARDCODED API KEY (replace with yours) ----------
OPENAI_API_KEY = ("USE YOUR OPEN-AI KEY)
client = OpenAI(api_key=OPENAI_API_KEY)


# """
# --------- PROMPT ----------
PROMPT = """
You are an expert in driver visual gaze behavior and traffic safety.

You will see four aligned images: RGB at t, gaze overlay at t, RGB at t+Δ, and gaze overlay at t+Δ.

Write EXACTLY FOUR SENTENCES (≤25 words each) following this structure:

(1) SCENE — Start with “The road scene shows …” or “The road has …”. Describe visible layout, lanes, intersections, vehicles, pedestrians, signals, and environment (day/night, weather). Mention only what is clearly visible.  
(2) CURRENT — Start with “The gaze is currently directed …” or “The gaze is focused …”. Identify the main fixation target(s) at time t (vehicle, signal, pedestrian, crosswalk, lane area, etc.).  
(3) NEXT — Start with “The gaze will …” (shift or remain). If the fixation stays on the same region, use “The gaze will remain …” or “No gaze shift expected.”  
(4) WHY — Start with “This fixation …” or “The gaze …”. Provide a safety-based rationale (≥20 words) explaining how it supports awareness, hazard anticipation, or compliance with road signals.

Constraints:
- Do NOT mention frames, overlays, Δ, or camera setup.
- Do NOT mention the ego vehicle; describe only visible scene elements and gaze targets.
- Use spatial or role-based references (e.g., “lead vehicle,” “pedestrian on right,” “traffic light ahead”).
- Avoid assumptions about hidden areas or unseen objects.
- Keep total text ≤100 words.

Return STRICT JSON only:
{
  "scene": "<one sentence>",
  "current": "<one sentence>",
  "next": "<one sentence>",
  "why": "<one sentence>",
  "caption": "SCENE. CURRENT. NEXT. WHY."
}
"""



# ---------- helpers: indexing & pairing ----------
FNAME_RE = re.compile(r"^(?P<vid>.+)_(?P<idx>\d{6})\.png$", re.IGNORECASE)

def list_pngs(dirpath: str) -> List[str]:
    return sorted([f for f in os.listdir(dirpath) if f.lower().endswith(".png")]) if os.path.isdir(dirpath) else []

def parse_name(fname: str) -> Optional[Tuple[str,int]]:
    m = FNAME_RE.match(fname)
    return (m.group("vid"), int(m.group("idx"))) if m else None

def index_dir(dirpath: str) -> Dict[str, Dict[int, str]]:
    out: DefaultDict[str, Dict[int, str]] = defaultdict(dict)
    for f in list_pngs(dirpath):
        p = parse_name(f)
        if p:
            vid, idx = p
            out[vid][idx] = os.path.join(dirpath, f)
    return out

def build_camera_overlay_index(rgb_dir: str, overlay_dir: str):
    return index_dir(rgb_dir), index_dir(overlay_dir)

def make_pairs_camera_next_within(cam_map, ovl_map, dmin=DELTA_MIN, dmax=DELTA_MAX):
    pairs = []
    for vid in sorted(cam_map.keys()):
        cam_idxs = sorted(cam_map[vid].keys())
        if not cam_idxs: continue
        ovl_idxs = set(ovl_map.get(vid, {}).keys())
        if not ovl_idxs: continue

        for i, t in enumerate(cam_idxs):
            tp = None
            for j in range(i+1, len(cam_idxs)):
                cand = cam_idxs[j]
                delta = cand - t
                if delta < dmin: continue
                if delta > dmax: break
                tp = cand
                break
            if tp is None: continue
            # require overlays at t and tp
            if (t not in ovl_idxs) or (tp not in ovl_idxs): continue

            pairs.append({
                "vid": vid,
                "t": t, "tp": tp, "delta": tp - t,
                "rgb_t": cam_map[vid][t],  "ovl_t": ovl_map[vid][t],
                "rgb_tp": cam_map[vid][tp],"ovl_tp": ovl_map[vid][tp],
                "key": f"{vid}_{t:06d}.png"
            })
    return pairs

# ---------- helpers: image to base64-jpeg ----------
def b64_jpg_small(path: str, target_w=JPEG_WIDTH, quality=JPEG_QUALITY) -> str:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    h, w = img.shape[:2]
    if w > target_w:
        img = cv2.resize(img, (target_w, int(round(h * (target_w / float(w))))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f"jpeg encode failed: {path}")
    return base64.b64encode(buf.tobytes()).decode()

# ---------- helpers: light cleaning (no rejection) ----------
NEG_PATTERNS = [
    r"\bno\s+visible\s+(pedestrians?|vehicles?|cars?|signals?|obstacles?)\b(?:[^.,;]*[.,;])?",
    r"\bnone\s+(are|is)\s+visible\b(?:[^.,;]*[.,;])?",
    r"\bnot\s+visible\b(?:[^.,;]*[.,;])?",
]

def _cap_first(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    return s[0].upper() + s[1:]


def strip_negatives(text: str) -> str:
    if not text: return ""
    s = " " + text.strip() + " "
    for pat in NEG_PATTERNS:
        s = re.sub(pat, "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"\s+([,;:])", r"\1", s)
    return s.strip()

def _normalize_no_shift(s: str) -> str:
    # unify to "No gaze shift expected."
    if re.search(r"\bno\s+gaze\s+shift\s+expected\b", (s or ""), flags=re.I):
        return "No gaze shift expected."
    return s

def _dehedge(s: str) -> str:
    # soften "may/might" to declarative wording for dataset consistency
    s = re.sub(r"\bmay\s+be\b", "is", (s or ""), flags=re.I)
    s = re.sub(r"\bmay\b", "will", s, flags=re.I)
    s = re.sub(r"\bmight\b", "will", s, flags=re.I)
    return s

def _trim_leadins(s: str) -> str:
    # e.g., “The scene shows …” → drop boilerplate
    return re.sub(r"^(The\s+scene\s+shows\s+|This\s+scene\s+shows\s+)", "", (s or ""), flags=re.I)

def clean_sentence(s: str, ensure_period=True, max_words=MAX_WORDS) -> str:
    s = (s or "").strip()
    s = s.replace("’", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
    s = _trim_leadins(s)
    s = _dehedge(s)
    s = _normalize_no_shift(s)
    if max_words:
        words = s.split()
        if len(words) > max_words:
            s = " ".join(words[:max_words])
    s = re.split(r'(?<=[.!?])\s+', s)[0].strip()
    if ensure_period and s and not s.endswith("."):
        s += "."
    s = _cap_first(s)            # <<--- add this
    return s


def ensure_gaze_shift(why: str) -> str:
    w = (why or "").strip()
    if "gaze shift" not in w.lower():
        w = w.rstrip(".") + ". This explains the gaze shift."
    return w

def clean_parsed(parsed: Dict) -> Dict:
    scene   = strip_negatives(parsed.get("scene", ""))
    current = parsed.get("current", "")
    next_s  = parsed.get("next", "")
    why     = parsed.get("why", "")

    scene   = clean_sentence(scene,   True, MAX_WORDS)
    current = clean_sentence(current, True, MAX_WORDS)
    next_s  = clean_sentence(next_s,  True, MAX_WORDS)
    why     = clean_sentence(ensure_gaze_shift(why), True, MAX_WORDS)

    caption = " ".join([scene, current, next_s, why]).strip()
    return {"scene": scene, "current": current, "next": next_s, "why": why, "caption": caption}

# ---------- robust JSON extraction ----------
def extract_json_blob(raw: str) -> Optional[str]:
    """Try to pull the first {...} JSON object from free-form text (handles code fences)."""
    if not raw: return None
    txt = raw.strip()
    if txt.startswith("```"):
        txt = re.sub(r"^```(json)?\s*|\s*```$", "", txt, flags=re.DOTALL).strip()
    # heuristic: find balanced braces
    start = txt.find("{")
    end = txt.rfind("}")
    if start != -1 and end != -1 and end > start:
        return txt[start:end+1]
    return None

def try_parse_json(raw: str) -> Optional[Dict]:
    blob = extract_json_blob(raw) or raw
    try:
        data = json.loads(blob)
        return data if isinstance(data, dict) else None
    except Exception:
        return None

# ---------- OpenAI call ----------
def call_gpt(imgs: List[str]) -> str:
    """
    Returns raw model text (expected JSON string). No response_format to avoid brittle failures.
    """
    parts = [{
        "type": "image_url",
        "image_url": {"url": f"data:image/jpeg;base64,{b64_jpg_small(p)}", "detail": "low"}
    } for p in imgs]
    messages = [{"role": "user", "content": parts + [{"type": "text", "text": PROMPT}]}]
    rsp = client.chat.completions.create(
        model=MODEL_ID,
        messages=messages,
        max_tokens=400,
        temperature=0.0
    )
    return (rsp.choices[0].message.content or "").strip()

def polite_sleep(rpm_target: float):
    # simple wall-clock throttle
    time.sleep(60.0 / max(1e-6, rpm_target))

# -------------- MAIN --------------
def main():
    os.makedirs(os.path.dirname(OUT_JSON) or ".", exist_ok=True)

    cam_map, ovl_map = build_camera_overlay_index(RGB_DIR, OVERLAY_DIR)
    if not cam_map or not ovl_map:
        print("[ERROR] Missing camera or overlay frames.")
        return

    pairs = make_pairs_camera_next_within(cam_map, ovl_map, DELTA_MIN, DELTA_MAX)
    if not pairs:
        print("[ERROR] No valid (t, t+Δ) pairs within 1..18. Check filenames.")
        return
    if LIMIT: pairs = pairs[:LIMIT]

    # Resume support
    out_records: List[Dict] = []
    done_keys = set()
    if os.path.isfile(RESUME_JSON):
        try:
            with open(RESUME_JSON, "r", encoding="utf-8") as f:
                prev = json.load(f)
                if isinstance(prev, list):
                    out_records = prev
                    done_keys = {r.get("image_t") for r in prev if isinstance(r, dict)}
                    done_keys = {k for k in done_keys if k}
        except Exception:
            pass

    cleaned_all = []   # final clean JSON
    csv_rows = []      # quick spreadsheet view

    for rec in tqdm(pairs, desc="Captioning"):
        key = rec["key"]
        if key in done_keys:
            continue

        imgs = [rec["rgb_t"], rec["ovl_t"], rec["rgb_tp"], rec["ovl_tp"]]
        if not all(os.path.isfile(p) for p in imgs):
            print(f"[skip] missing files for {key}")
            continue

        raw_txt = ""
        for attempt in range(2):  
            try:
                raw_txt = call_gpt(imgs)
                break
            except Exception as e:
                raw_txt = f"{type(e).__name__}: {e}"
            polite_sleep(RPM_TARGET)

        parsed = try_parse_json(raw_txt)
        clean = {"scene":"", "current":"", "next":"", "why":"", "caption":""}
        if isinstance(parsed, dict):
            clean = clean_parsed(parsed)

        # Keep a progress record with both raw and clean
        item = {
            "prompt_version": PROMPT_VERSION,
            "image_t": os.path.basename(rec["rgb_t"]),
            "image_tp": os.path.basename(rec["rgb_tp"]),
            "delta": rec["delta"],
            "raw": raw_txt,
            "scene": clean["scene"],
            "current": clean["current"],
            "next": clean["next"],
            "why": clean["why"],
            "caption": clean["caption"]
        }
        out_records.append(item)

        # Append to final clean set & CSV
        cleaned_all.append({
            "prompt_version": PROMPT_VERSION,
            "image_t": item["image_t"],
            "image_tp": item["image_tp"],
            "delta": item["delta"],
            "scene": item["scene"],
            "current": item["current"],
            "next": item["next"],
            "why": item["why"],
            "caption": item["caption"]
        })
        csv_rows.append([item["image_t"], item["image_tp"], item["delta"], item["caption"]])

        # incremental save
        with open(RESUME_JSON, "w", encoding="utf-8") as f:
            json.dump(out_records, f, ensure_ascii=False, indent=2)

        polite_sleep(RPM_TARGET)

    # final save
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(cleaned_all, f, ensure_ascii=False, indent=2)

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_t","image_tp","delta","caption"])
        w.writerows(csv_rows)

    print(f"\nSaved {len(cleaned_all)} items:")
    print(f"  JSON -> {OUT_JSON}")
    print(f"  CSV  -> {OUT_CSV}")
    print(f"  Progress (raw+clean) -> {RESUME_JSON}")

if __name__ == "__main__":
    main()
