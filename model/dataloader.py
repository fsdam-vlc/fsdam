# config.py 
from pathlib import Path
import torch

THIS_DIR  = Path(__file__).resolve().parent
PROJ_ROOT = THIS_DIR


class CFG:
    # ===== Experiment =====
    EXP_NAME = "fsdam"

    # ===== Data =====
    TRAIN_RGB_DIR  = (PROJ_ROOT / "few_shot_dataset" / "caption_gaze" / "train" / "camera").resolve()
    TRAIN_GAZE_DIR = (PROJ_ROOT / "few_shot_dataset" / "caption_gaze" / "train" / "gaze").resolve()
    VAL_RGB_DIR    = (PROJ_ROOT / "few_shot_dataset" / "caption_gaze" / "val"   / "camera").resolve()
    VAL_GAZE_DIR   = (PROJ_ROOT / "few_shot_dataset" / "caption_gaze" / "val"   / "gaze").resolve()
    TRAIN_JSON     = (PROJ_ROOT / "few_shot_dataset" / "caption_gaze" / "bdda_train.json").resolve()
    VAL_JSON       = (PROJ_ROOT / "few_shot_dataset" / "caption_gaze" / "bdda_val.json").resolve()

    # ===== Prompt =====
    FIXED_Q = (
        "Answer the following for the image: What is in the scene, where is the driver "
        "looking now, and where will the gaze shift next and why?\n"
        "Return exactly four sentences in this order: "
        "WHAT (scene summary). WHERE-NOW (current driver gaze target). "
        "WHERE-NEXT (next gaze target). WHY (reason for the shift).\n"
        "Do not use lists or numbering."
    )

    # ===== Geometry / dims =====
    GAZE_OUT  = 64
    GRID_TOK  = 24
    D_COMMON  = 768

    # ===== Backbone =====
    LLAVA_MODEL = "llava-hf/llava-v1.6-mistral-7b-hf"

    # ===== LoRA on LM =====
    LORA_R        = 16
    LORA_ALPHA    = 32
    # LORA_DROPOUT  = 0.2
    # LORA_TARGET   = ["q_proj", "k_proj", "v_proj", "o_proj"]
    LORA_DROPOUT    = 0.2
    LORA_TARGET     = ["q_proj", "k_proj", "v_proj", "o_proj"]
    # ===== GCLA and caption prefix =====
    GCLA_HEADS      = 16
    M_GTOK          = 16
    M_GTOK_GAZE     = 16
    PREFIX_TOKENS   = 8

    # ===== Gaze Head =====
    GAZE_TAU = 1.2

    # ===== Blur gap gaze loss =====
    BG_SIGMA  = 1.0
    BG_LAMBDA = 0.3
    BG_MARGIN = 0.05

    # ===== Loss weights =====
    WEIGHTS = dict(
        gaze = 1.0,
        align= 0.2,
        cap  = 1.0,
        align_start = 0.0,
    )
    ALIGN_TAU  = 0.07
    RAMP_EPOCH = 0

    # ===== Training =====
    BATCH_SIZE   = 4
    NUM_WORKERS  = 4
    EPOCHS       = 12
    MIXED_PREC   = "bf16"
    GRAD_ACCUM   = 4
    LR_LORA      = 1e-4
    LR_HEADS     = 2e-4
    GRAD_CLIP    = 1.0

    # ===== Inference and eval =====
    EVAL_BEAMS            = 3
    EVAL_MAX_NEW_TOKENS   = 128
    REPETITION_PENALTY    = 0.0
    LENGTH_PENALTY        = 0.0

    # ===== Runtime =====
    DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
    SEED         = 42

    # ===== Checkpoints =====
    CKPT_DIR = (PROJ_ROOT / "checkpoints" / EXP_NAME).resolve()
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    # ===== Ablation toggles =====
    # GCLA: off, shared, dual
    GCLA_MODE        = "dual"
    GCLA_DROPOUT     = 0.0
    GCLA_PRENORM     = False
    GCLA_RESIDUAL    = False
    GCLA_POSENC      = "none"     # none, sine2d
    GCLA_KV_CONV1x1  = False
    # Heads and queries are above: GCLA_HEADS, M_GTOK, M_GTOK_GAZE

    # Branch toggles
    CAPTION_ON       = True
    ALIGN_ON         = True
    GAZE_BLUR_GAP_ON = True
    PREFIX_ON        = True
    LORA_ON          = True
    GAZE_TAU_LEARN   = False    # learnable temperature in gaze head


def sanity_check():
    paths = [
        CFG.TRAIN_RGB_DIR, CFG.TRAIN_GAZE_DIR,
        CFG.VAL_RGB_DIR,   CFG.VAL_GAZE_DIR,
        CFG.TRAIN_JSON,    CFG.VAL_JSON,
        CFG.CKPT_DIR,
    ]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing paths:\n" + "\n".join(" - " + m for m in missing))
