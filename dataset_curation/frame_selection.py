#!/usr/bin/env python3
# FSDAM: Pure KL–Δ frame pairing 
# For each video:
#   1) anchors t = peaks of smoothed KL(g_t || g_{t-1})
#   2) pick Δ* in [DELTA_MIN, DELTA_MAX] maximizing KL(g_{t+Δ} || g_t)
#   3) keep top-K_SELECT anchors w/ temporal spacing
#   4) export: scene_t, gaze_t, scene_{t+Δ*}, gaze_{t+Δ*}
# for bdd-a dataset to fsdam few shot dataset
import os, re, cv2, numpy as np
from tqdm import tqdm

# ========= CONFIG =========
#edit the path at according to your directory
CAMERA_ROOT = "dataset/BDDA/test/camera_videos"      # RGB videos
GAZE_ROOT   = "dataset/BDDA/test/gazemap_videos"     # Gaze videos
OUT_RGB_DIR = "dataset/BDDA/test/preprocessed/camera"
OUT_GAZ_DIR = "dataset/BDDA/test/preprocessed/gaze"

# Selection policy (fixed-K per video)
K_SELECT     = 2        # pairs per video (use 2–4 for mining; 2 for val/test is fine)
DELTA_MIN    = 3        # frames (~0.1s @30fps)
DELTA_MAX    = 18       # frames (~0.6s @30fps)
MIN_SPACING  = 25       # frames between chosen anchors (diversity)
SMOOTH_WIN   = 5        # smoothing window over KL(g_t || g_{t-1})
MIN_LEN      = 50       # skip very short videos

# Geometry
CROP_H, CROP_W   = 576, 1280   # center-crop gaze before histogram
EXPORT_H, EXPORT_W = 720, 1280 # saved image size

# Misc
GLOBAL_MAX = 10**9      # global cap on (t, t+Δ*) units saved
SEED = 0                # determinism

# ========= Reproducibility =========
np.random.seed(SEED)
cv2.setNumThreads(1)

# ========= Utils =========
def ensure_dir(p): 
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def list_videos(root):
    exts = {".mp4",".avi",".mov",".mkv"}
    out = {}
    for f in sorted(os.listdir(root)):
        full = os.path.join(root, f)
        stem, ext = os.path.splitext(f)
        if os.path.isfile(full) and ext.lower() in exts:
            try:
                if os.path.getsize(full) > 0:
                    out[stem] = full
            except OSError:
                pass
    return out

def norm_cam(stem): 
    return stem

def norm_gaze(stem):  # e.g., 2_pure_hm -> 2
    return re.sub(r'(_pure_hm|_hm|_pure|_gaze(map)?|_heat(map)?)$', '', stem, flags=re.IGNORECASE)

def center_crop_gray(img_bgr, ch=CROP_H, cw=CROP_W):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = g.shape
    y0, x0 = max(0,(H-ch)//2), max(0,(W-cw)//2)
    return g[y0:y0+ch, x0:x0+cw]

def hist256(gray):
    return cv2.calcHist([gray.astype(np.uint8)], [0], None, [256], [0,256]).flatten().astype(np.float32)

def safe_norm(h, eps=1e-8): 
    return h / (h.sum() + eps)

def kl_div(p, q, eps=1e-8):
    p = safe_norm(p, eps); q = safe_norm(q, eps)
    return float(np.sum(p * (np.log(p + eps) - np.log(q + eps))))

def smooth1d(x, win=SMOOTH_WIN):
    if win <= 1: return x
    k = np.ones(win, np.float32)/win
    return np.convolve(x, k, mode="same")

def nms_1d(scores, radius, topk):
    order = np.argsort(scores)[::-1]
    used = np.zeros_like(scores, dtype=bool)
    keep = []
    for i in order:
        if used[i]: continue
        keep.append(i)
        lo, hi = max(0,i-radius), min(len(scores), i+radius+1)
        used[lo:hi] = True
        if len(keep) >= topk: break
    return sorted(keep)

# ========= Core =========
def mine_pairs_one_video(vid, cam_path, gaze_path):
    gz_cap = cv2.VideoCapture(gaze_path)
    if not gz_cap.isOpened(): 
        return []

    T = int(gz_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if T < MIN_LEN:
        gz_cap.release(); 
        return []

    # Read all gaze frames once 
    gaze_gray, gaze_hist = [], []
    for _ in range(T):
        ok, f = gz_cap.read()
        if not ok: break
        gg = center_crop_gray(f)
        gaze_gray.append(gg)
        gaze_hist.append(hist256(gg))
    gz_cap.release()

    T = len(gaze_hist)
    if T < MIN_LEN: 
        return []

    # Anchor curve: KL(g_t || g_{t-1})
    kl_prev = np.array([kl_div(gaze_hist[t], gaze_hist[t-1]) for t in range(1, T)], np.float32)
    if kl_prev.size == 0: 
        return []

    scores  = smooth1d(kl_prev, SMOOTH_WIN)                # length T-1
    anchors = nms_1d(scores, MIN_SPACING, max(8, 4*K_SELECT))
    if not anchors: 
        return []

    # For each anchor t, pick tp in [t+Δmin, t+Δmax] maximizing KL(g_tp || g_t)
    picks = []
    for a in anchors:
        t = max(1, a)                                      # ensure t>=1
        upper = min(DELTA_MAX, T-1 - t)
        if upper < DELTA_MIN: 
            continue
        h_t = gaze_hist[t]
        best_tp, best = None, -1e9
        for d in range(DELTA_MIN, upper+1):
            tp = t + d
            score = kl_div(gaze_hist[tp], h_t)             # future vs current
            if score > best:
                best, best_tp = score, tp
        if best_tp is not None:
            picks.append((t, best_tp, best))

    if not picks: 
        return []

    # Sort by score; enforce spacing on anchors; keep top-K_SELECT
    picks.sort(key=lambda x: x[2], reverse=True)
    kept = []
    for (t,tp,_) in picks:
        if all(abs(t - kt) >= MIN_SPACING for (kt,_) in kept):
            kept.append((t,tp))
            if len(kept) >= K_SELECT: 
                break
    return kept

def export_pairs(vid, cam_path, gaze_path, pairs, counter):
    if not pairs: 
        return counter

    cam = cv2.VideoCapture(cam_path)
    gz  = cv2.VideoCapture(gaze_path)
    if not cam.isOpened() or not gz.isOpened():
        cam.release(); gz.release(); 
        return counter

    ensure_dir(OUT_RGB_DIR); ensure_dir(OUT_GAZ_DIR)

    for (t,tp) in pairs:
        if counter >= GLOBAL_MAX: break

        # --- frame t ---
        cam.set(cv2.CAP_PROP_POS_FRAMES, t)
        gz.set(cv2.CAP_PROP_POS_FRAMES, t)
        ok1, rgb_t = cam.read()
        ok2, g_t   = gz.read()
        if not (ok1 and ok2): 
            continue

        rgb_t = cv2.resize(rgb_t, (EXPORT_W, EXPORT_H))
        g_t   = cv2.resize(cv2.cvtColor(g_t, cv2.COLOR_BGR2GRAY), (EXPORT_W, EXPORT_H))

        name_t = f"{vid}_{t:06d}.png"
        cv2.imwrite(os.path.join(OUT_RGB_DIR, name_t), rgb_t)
        cv2.imwrite(os.path.join(OUT_GAZ_DIR, name_t), g_t)

        # --- frame t+Δ* ---
        cam.set(cv2.CAP_PROP_POS_FRAMES, tp)
        gz.set(cv2.CAP_PROP_POS_FRAMES, tp)
        ok3, rgb_tp = cam.read()
        ok4, g_tp   = gz.read()
        if ok3 and ok4:
            rgb_tp = cv2.resize(rgb_tp, (EXPORT_W, EXPORT_H))
            g_tp   = cv2.resize(cv2.cvtColor(g_tp, cv2.COLOR_BGR2GRAY), (EXPORT_W, EXPORT_H))
            name_tp = f"{vid}_{tp:06d}.png"
            cv2.imwrite(os.path.join(OUT_RGB_DIR, name_tp), rgb_tp)
            cv2.imwrite(os.path.join(OUT_GAZ_DIR, name_tp), g_tp)

        counter += 1

    cam.release(); gz.release()
    return counter

def main():
    ensure_dir(OUT_RGB_DIR); ensure_dir(OUT_GAZ_DIR)

    cam_map_raw  = list_videos(CAMERA_ROOT)
    gaze_map_raw = list_videos(GAZE_ROOT)
    cam_map  = {norm_cam(k): v for k,v in cam_map_raw.items()}
    gaze_map = {norm_gaze(k): v for k,v in gaze_map_raw.items()}
    vids = sorted(set(cam_map.keys()) & set(gaze_map.keys()))
    if not vids:
        print("[ERROR] No overlapping videos between CAMERA_ROOT and GAZE_ROOT.")
        return

    total_pairs, skipped, counter = 0, 0, 0
    for vid in tqdm(vids, desc="Curating"):
        if counter >= GLOBAL_MAX: break
        pairs = mine_pairs_one_video(vid, cam_map[vid], gaze_map[vid])
        if not pairs:
            skipped += 1; 
            continue
        counter = export_pairs(vid, cam_map[vid], gaze_map[vid], pairs, counter)
        total_pairs += len(pairs)

    print("\n=== Summary ===")
    print(f"Videos: {len(vids)} | Skipped: {skipped} | Pairs mined: {total_pairs}")
    print(f"RGB out : {OUT_RGB_DIR}")
    print(f"Gaze out: {OUT_GAZ_DIR}  (each pair writes t and t+Δ*)")

if __name__ == "__main__":
    main()
