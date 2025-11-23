# architecture.py

# this is also toggle aware for doing ablation trhough config.py
from typing import List, Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from peft import get_peft_model, LoraConfig

from config import CFG


# ===================== Utils =====================
def info_nce(u_a: torch.Tensor, u_b: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    u_a = F.normalize(u_a, dim=-1)
    u_b = F.normalize(u_b, dim=-1)
    logits = u_a @ u_b.t() / tau
    labels = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, labels)

def spatial_softmax(logits: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    B, C, H, W = logits.shape
    s = (logits.view(B, 1, -1) / max(tau, 1e-6)).softmax(dim=-1)
    return s.view(B, 1, H, W)


# ===================== Blur gap gaze loss =====================
def _gaussian_kernel1d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    k = int(2 * math.ceil(3 * float(max(sigma, 1e-3))) + 1)
    x = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2.0
    w = torch.exp(-(x ** 2) / (2 * sigma * sigma))
    w = w / w.sum().clamp_min(1e-8)
    return w.view(1, 1, k)

def _blur2d_prob_map(q: torch.Tensor, sigma: float) -> torch.Tensor:
    B, C, H, W = q.shape
    device, dtype = q.device, q.dtype
    k1d = _gaussian_kernel1d(sigma, device, dtype)
    ky = k1d.view(1, 1, -1, 1)
    kx = k1d.view(1, 1, 1, -1)
    q_y = F.conv2d(q, ky, padding=(ky.shape[2] // 2, 0), groups=1)
    q_blur = F.conv2d(q_y, kx, padding=(0, kx.shape[3] // 2), groups=1)
    q_blur = q_blur.clamp_min(0)
    q_blur = q_blur / q_blur.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-8)
    return q_blur

def gaze_loss_blur_gap(
    p_gt: torch.Tensor,
    q_pred: torch.Tensor,
    sigma: float = CFG.BG_SIGMA,
    lam:   float = CFG.BG_LAMBDA,
    margin: float = CFG.BG_MARGIN,
    eps:   float = 1e-6,
):
    p = p_gt / p_gt.sum(dim=(1,2,3), keepdim=True).clamp_min(1e-8)
    q = q_pred / q_pred.sum(dim=(1,2,3), keepdim=True).clamp_min(1e-8)
    q_blur = _blur2d_prob_map(q, sigma)
    kl_sharp = (p.clamp_min(eps) * (p.clamp_min(eps).log() - q.clamp_min(eps).log())).sum(dim=(1,2,3))
    kl_blur  = (p.clamp_min(eps) * (p.clamp_min(eps).log() - q_blur.clamp_min(eps).log())).sum(dim=(1,2,3))
    gap = F.relu(kl_blur - kl_sharp + margin)
    loss = kl_sharp.mean() + lam * gap.mean()
    logs = {
        "kl_sharp": float(kl_sharp.mean().item()),
        "kl_blur":  float(kl_blur.mean().item()),
        "gap_pos":  float(gap.mean().item()),
    }
    return loss, logs

def gaze_loss_kl_only(p_gt: torch.Tensor, q_pred: torch.Tensor, eps: float = 1e-6):
    p = p_gt / p_gt.sum(dim=(1,2,3), keepdim=True).clamp_min(1e-8)
    q = q_pred / q_pred.sum(dim=(1,2,3), keepdim=True).clamp_min(1e-8)
    kl = (p.clamp_min(eps) * (p.clamp_min(eps).log() - q.clamp_min(eps).log())).sum(dim=(1,2,3))
    return kl.mean()


# ===================== Positional enc =====================
def sine2d_posenc(H: int, W: int, dim: int, device: torch.device) -> torch.Tensor:
    # standard 2D sin-cos pe, dim must be even
    assert dim % 4 == 0, "sine2d needs dim divisible by 4"
    y = torch.arange(H, device=device).float()
    x = torch.arange(W, device=device).float()
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    omega = torch.arange(dim // 4, device=device).float()
    omega = 1.0 / (10000 ** (omega / (dim // 4)))
    sin_x = torch.sin(xx[..., None] * omega[None, None, :])
    cos_x = torch.cos(xx[..., None] * omega[None, None, :])
    sin_y = torch.sin(yy[..., None] * omega[None, None, :])
    cos_y = torch.cos(yy[..., None] * omega[None, None, :])
    pe = torch.cat([sin_x, cos_x, sin_y, cos_y], dim=-1)  # [H,W,dim]
    return pe.view(H * W, dim)


# ===================== GCLA =====================
class GCLACrossAttention(nn.Module):
    def __init__(self, d_lm: int, d_mem: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.d_lm    = d_lm
        self.d_mem   = d_mem
        self.n_heads = n_heads

        self.q_ln = nn.LayerNorm(d_lm) if CFG.GCLA_PRENORM else nn.Identity()
        self.kv_ln = nn.LayerNorm(d_mem) if CFG.GCLA_PRENORM else nn.Identity()

        self.kv_conv = nn.Conv2d(d_mem, d_mem, 1) if CFG.GCLA_KV_CONV1x1 else nn.Identity()

        self.q_proj = nn.Linear(d_lm, d_lm, bias=False)
        self.k_proj = nn.Linear(d_mem, d_lm, bias=False)
        self.v_proj = nn.Linear(d_mem, d_lm, bias=False)
        self.out    = nn.Linear(d_lm, d_lm, bias=False)
        self.drop   = nn.Dropout(dropout)
        self._dbg_once = False

    def forward(self, q_text: torch.Tensor, fmap: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if q_text.dim() == 2:
            q_text = q_text.unsqueeze(1)
        B, M, _ = q_text.shape
        Bv, d_mem, H, W = fmap.shape
        assert Bv == B and H == CFG.GRID_TOK and W == CFG.GRID_TOK
        fmap2 = self.kv_conv(fmap)
        # flatten to [B,S,d_mem]
        mem = fmap2.flatten(2).transpose(1, 2)
        # optional sine2d pos enc
        if CFG.GCLA_POSENC == "sine2d":
            pe = sine2d_posenc(H, W, d_mem, mem.device)  # [S,d_mem]
            mem = mem + pe[None, :, :]

        mem = self.kv_ln(mem)
        k = self.k_proj(mem)
        v = self.v_proj(mem)

        q = self.q_proj(self.q_ln(q_text)).reshape(B * M, self.d_lm)
        h = self.n_heads
        d = self.d_lm // h

        q = q.view(B * M, h, d).unsqueeze(2)                                  # [BM,h,1,d]
        k = k.repeat_interleave(M, 0).view(B * M, -1, h, d).transpose(1, 2)   # [BM,h,S,d]
        v = v.repeat_interleave(M, 0).view(B * M, -1, h, d).transpose(1, 2)   # [BM,h,S,d]

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(d)
        attn = attn.softmax(dim=-1)
        ctx  = attn @ v
        ctx  = ctx.transpose(1, 2).reshape(B * M, self.d_lm)
        ctx  = self.drop(self.out(ctx))
        ctx_out = ctx.view(B, M, self.d_lm)
        if CFG.GCLA_RESIDUAL:
            ctx_out = ctx_out + q_text

        attn_sp = attn.mean(dim=1).squeeze(2).view(B, M, -1)
        if not self._dbg_once:
            print(f"[GCLA] q_text={tuple(q_text.shape)} fmap={tuple(fmap.shape)} attn={tuple(attn_sp.shape)}")
            self._dbg_once = True
        return ctx_out, attn_sp


# ===================== LLaVA Backbone =====================
class LlavaBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.processor: LlavaNextProcessor = LlavaNextProcessor.from_pretrained(
            CFG.LLAVA_MODEL, use_fast=False
        )
        self.model: LlavaNextForConditionalGeneration = LlavaNextForConditionalGeneration.from_pretrained(
            CFG.LLAVA_MODEL,
            torch_dtype=torch.bfloat16 if (CFG.MIXED_PREC == "bf16" and torch.cuda.is_available()) else torch.float32,
            device_map=None
        )
        self.device = torch.device(CFG.DEVICE)
        self.model.to(self.device)

        # Freeze backbone
        for p in self.model.parameters():
            p.requires_grad_(False)

        # LoRA on LM
        if CFG.LORA_ON:
            lm = self.model.language_model if hasattr(self.model, "language_model") else self.model.model.language_model
            lcfg = LoraConfig(
                r=CFG.LORA_R, lora_alpha=CFG.LORA_ALPHA, lora_dropout=CFG.LORA_DROPOUT,
                bias="none", task_type="CAUSAL_LM", target_modules=CFG.LORA_TARGET
            )
            self.model.language_model = get_peft_model(lm, lcfg)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.model.parameters())
        print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {100*trainable/total:.4f}")

        self._d_vlm, self._grid = self._probe_vision_dim_and_grid()
        self._d_lm  = int(self.model.config.text_config.hidden_size)

    @property
    def d_vlm(self) -> int: return self._d_vlm
    @property
    def d_lm(self) -> int:  return self._d_lm
    @property
    def grid(self) -> int:  return self._grid

    @torch.no_grad()
    def _probe_vision_dim_and_grid(self) -> Tuple[int, int]:
        dummy = Image.new("RGB", (336, 336), (255, 255, 255))
        ip = self.processor.image_processor
        batch = ip(images=[dummy], return_tensors="pt")
        pix = batch["pixel_values"].to(self.device, dtype=self.model.dtype)
        vt  = self.model.vision_tower
        B, P, C, H, W = pix.shape
        out = vt(pixel_values=pix.view(B*P, C, H, W), output_hidden_states=True, return_dict=True)
        feats = out.last_hidden_state
        if feats.size(1) > 1:
            feats = feats[:, 1:, :]
        D = int(feats.shape[-1])
        S = int(feats.shape[1])
        g = int(math.sqrt(S))
        print(f"[probe] grid={g} d_vlm={D}")
        return D, g

    @torch.no_grad()
    def encode(self, images: List[Image.Image]) -> Tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        images_336 = [img.convert("RGB").resize((336, 336), Image.Resampling.BICUBIC) for img in images]
        ip = self.processor.image_processor
        batch = ip(images=images_336, return_tensors="pt")
        pixel_values = batch["pixel_values"].to(self.device, dtype=self.model.dtype)
        vt = self.model.vision_tower

        B, P, C, H, W = pixel_values.shape
        pv = pixel_values.view(B*P, C, H, W)
        vt_out = vt(pixel_values=pv, output_hidden_states=True, return_dict=True)
        feats = vt_out.last_hidden_state
        feats_no_cls = feats[:, 1:, :] if feats.size(1) > 1 else feats

        pooled_tok = feats_no_cls.mean(dim=1)
        global_emb = pooled_tok.view(B, P, -1).mean(dim=1)

        D = feats_no_cls.shape[-1]
        S = feats_no_cls.shape[1]
        g = self._grid
        assert S == g * g
        fmap = feats_no_cls.view(B, P, g, g, D).mean(dim=1)
        fmap = fmap.permute(0, 3, 1, 2).contiguous()
        return global_emb.to(torch.float32), fmap.to(torch.float32)

    def _build_mm_batch(self, images: List[Image.Image], prompts: List[str],
                        add_generation_prompt: bool = False):
        msgs = []
        for q in prompts:
            q_clean = q.replace("<image>", "").strip()
            msgs.append([{"role": "user",
                          "content": [{"type": "image"},
                                      {"type": "text", "text": q_clean}]}])

        texts = self.processor.apply_chat_template(
            msgs, add_generation_prompt=add_generation_prompt, tokenize=False
        )
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"

        inputs = self.processor(images=images, text=texts, return_tensors="pt", padding=True)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                if k == "pixel_values":
                    inputs[k] = v.to(self.device, dtype=self.model.dtype)
                else:
                    inputs[k] = v.to(self.device)
        return inputs

    @torch.no_grad()
    def generate(self, images: List[Image.Image], prompts: List[str], **gen_kwargs) -> List[str]:
        self.model.eval()
        inputs = self._build_mm_batch(images, prompts, add_generation_prompt=True)
        gen_kwargs.setdefault("max_new_tokens", 120)
        gen_kwargs.setdefault("do_sample", False)
        out_ids = self.model.generate(**inputs, **gen_kwargs)
        texts = self.processor.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        clean = []
        for t in texts:
            clean.append(t.split("[/INST]")[-1].strip() if "[/INST]" in t else t.strip())
        return clean

    def caption_ce_with_prefix(self, images: List[Image.Image], prompts: List[str],
                               answers: List[str], prefix: torch.Tensor) -> torch.Tensor:
        self.model.train()
        dev = self.device
        inputs = self._build_mm_batch(images, prompts, add_generation_prompt=True)

        tok = self.processor.tokenizer
        tok_emb = self.model.language_model.get_input_embeddings()
        prompt_ids = inputs["input_ids"]
        prompt_emb = tok_emb(prompt_ids)
        B, Lp, D = prompt_emb.shape

        if CFG.PREFIX_ON:
            assert prefix.size(0) == B and prefix.size(2) == D
            inputs_embeds_prompt = torch.cat([prefix.to(prompt_emb.dtype).to(dev), prompt_emb], dim=1)
        else:
            inputs_embeds_prompt = prompt_emb

        ans_tok = tok(answers, return_tensors="pt", padding=True, add_special_tokens=False)
        input_ids_ans = ans_tok["input_ids"].to(dev)
        attn_mask_ans = ans_tok.get("attention_mask", None)
        if attn_mask_ans is not None:
            attn_mask_ans = attn_mask_ans.to(dev)

        ans_emb = tok_emb(input_ids_ans)
        inputs_embeds = torch.cat([inputs_embeds_prompt, ans_emb], dim=1)

        Lpfx = prefix.size(1) if CFG.PREFIX_ON else 0

        def _get_ctx_len():
            lm = getattr(self.model, "language_model", None) or getattr(self.model, "model", None)
            if lm is not None and hasattr(lm, "config"):
                v = getattr(lm.config, "max_position_embeddings", None)
                if isinstance(v, int) and v > 0:
                    return v
            tc = getattr(self.model.config, "text_config", None)
            if tc is not None:
                v = getattr(tc, "max_position_embeddings", None)
                if isinstance(v, int) and v > 0:
                    return v
            v = getattr(self.processor.tokenizer, "model_max_length", None)
            if isinstance(v, int) and 0 < v < 10**9:
                return v
            return 4096

        Lmax = int(_get_ctx_len())
        Lall = inputs_embeds.size(1)
        if Lall > Lmax:
            trim = Lall - Lmax
            inputs_embeds_prompt = inputs_embeds_prompt[:, trim:, :]
            inputs_embeds = torch.cat([inputs_embeds_prompt, ans_emb], dim=1)
            Lp = inputs_embeds_prompt.size(1) - Lpfx
        else:
            Lp = prompt_emb.size(1)

        Lall = inputs_embeds.size(1)
        attn_mask = torch.ones((B, Lall), dtype=torch.long, device=dev)

        labels = torch.full((B, Lall), fill_value=-100, device=dev)
        if attn_mask_ans is not None:
            ans_len_each = attn_mask_ans.sum(dim=1).long()
        else:
            ans_len_each = torch.full((B,), input_ids_ans.size(1), device=dev, dtype=torch.long)
        for b in range(B):
            la = int(ans_len_each[b].item())
            labels[b, (Lpfx + Lp):(Lpfx + Lp + la)] = input_ids_ans[b, :la]

        out = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            pixel_values=inputs["pixel_values"],
            image_sizes=inputs.get("image_sizes", None),
            labels=labels
        )
        return out.loss


# ===================== Heads and adapters =====================
class GazeHead(nn.Module):
    def __init__(self, d_in: int, out_res: int = CFG.GAZE_OUT, tau: float = CFG.GAZE_TAU):
        super().__init__()
        self.tau_param = nn.Parameter(torch.tensor(float(tau))) if CFG.GAZE_TAU_LEARN else None
        self.tau = tau
        self.up = nn.Sequential(
            nn.Conv2d(d_in, 256, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.10),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.10),
            nn.Upsample(scale_factor=4/3, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.05),
            nn.Conv2d(128, 1, 3, padding=1),
        )

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        logits = self.up(fmap)
        tau = self.tau_param.exp().item() if self.tau_param is not None else self.tau
        probs  = spatial_softmax(logits, tau=tau)
        return probs

class ProjHead(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_out), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(d_out, d_out), nn.Dropout(0.2),
        )
    def forward(self, x): return self.net(x)

class PrefixAdapter(nn.Module):
    def __init__(self, d_lm: int, L_p: int):
        super().__init__()
        self.L_p = L_p
        self.proj = nn.Sequential(
            nn.Linear(d_lm, d_lm),
            nn.Tanh(),
            nn.Linear(d_lm, d_lm * L_p),
        )
    def forward(self, ctx_bar: torch.Tensor) -> torch.Tensor:
        B, D = ctx_bar.shape
        out = self.proj(ctx_bar)
        return out.view(B, self.L_p, D)


# ===================== Joint Model =====================
class JointModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm = LlavaBackbone()

        # Gaze head
        self.gaze_head = GazeHead(self.vlm.d_vlm, CFG.GAZE_OUT, CFG.GAZE_TAU)

        # GCLA variants
        dropout = CFG.GCLA_DROPOUT
        if CFG.GCLA_MODE in ("dual", "shared"):
            self.gcla_cap  = GCLACrossAttention(self.vlm.d_lm, self.vlm.d_vlm, n_heads=CFG.GCLA_HEADS, dropout=dropout)
        if CFG.GCLA_MODE == "dual":
            self.gcla_gaze = GCLACrossAttention(self.vlm.d_lm, self.vlm.d_vlm, n_heads=CFG.GCLA_HEADS, dropout=dropout)

        # Query makers
        self.text2multi_cap  = nn.Linear(self.vlm.d_lm, self.vlm.d_lm * CFG.M_GTOK)
        self.text2multi_gaze = nn.Linear(self.vlm.d_lm, self.vlm.d_lm * CFG.M_GTOK_GAZE)

        # Projections
        self.proj_vis  = ProjHead(self.vlm.d_vlm, CFG.D_COMMON)
        self.proj_txt  = ProjHead(self.vlm.d_lm,  CFG.D_COMMON)

        # Caption prefix
        self.prefix_adapter = PrefixAdapter(self.vlm.d_lm, CFG.PREFIX_TOKENS)

        # Loss weights
        self.loss_w = dict(CFG.WEIGHTS)

        self.current_epoch = 1
        print(f"[LLaVA] d_vlm={self.vlm.d_vlm} d_lm={self.vlm.d_lm} grid={self.vlm.grid} gcla_heads={CFG.GCLA_HEADS}")
        self.to(CFG.DEVICE)
        self._dbg_once = False

    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)

    def set_loss_weights(self, w: Dict[str, float]):
        if not w:
            return
        for k, v in w.items():
            if k in self.loss_w and v is not None:
                self.loss_w[k] = float(v)

    # ===== helper to apply GCLA according to mode =====
    def _run_gcla(self, branch: str, q_text: torch.Tensor, fmap: torch.Tensor) -> torch.Tensor:
        if CFG.GCLA_MODE == "off":
            # simple pass through
            if branch == "cap":
                M = CFG.M_GTOK
            else:
                M = CFG.M_GTOK_GAZE
            return q_text.unsqueeze(1).repeat(1, M, 1)
        elif CFG.GCLA_MODE == "shared":
            gcla = self.gcla_cap
            if branch == "cap":
                q = self.text2multi_cap(q_text).view(q_text.size(0), CFG.M_GTOK, self.vlm.d_lm)
            else:
                q = self.text2multi_gaze(q_text).view(q_text.size(0), CFG.M_GTOK_GAZE, self.vlm.d_lm)
            ctx, _ = gcla(q.to(fmap.dtype), fmap)
            return ctx
        else:
            if branch == "cap":
                q = self.text2multi_cap(q_text).view(q_text.size(0), CFG.M_GTOK, self.vlm.d_lm)
                ctx, _ = self.gcla_cap(q.to(fmap.dtype), fmap)
            else:
                q = self.text2multi_gaze(q_text).view(q_text.size(0), CFG.M_GTOK_GAZE, self.vlm.d_lm)
                ctx, _ = self.gcla_gaze(q.to(fmap.dtype), fmap)
            return ctx

    def forward_train(self, batch: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        images  = batch["images"]
        prompts = batch["prompts"]
        answers = batch["answers"]
        gaze_gt = batch["gaze"].to(CFG.DEVICE)

        # Vision
        with torch.no_grad():
            _, fmap = self.vlm.encode(images)

        # Gaze pred
        pred_gaze = self.gaze_head(fmap)

        # Gaze loss
        if CFG.GAZE_BLUR_GAP_ON:
            loss_gaze, glogs = gaze_loss_blur_gap(
                p_gt=gaze_gt, q_pred=pred_gaze,
                sigma=CFG.BG_SIGMA, lam=CFG.BG_LAMBDA, margin=CFG.BG_MARGIN
            )
        else:
            loss_gaze = gaze_loss_kl_only(gaze_gt, pred_gaze)
            glogs = {"kl_sharp": float("nan"), "kl_blur": float("nan"), "gap_pos": float("nan")}

        # Text context
        q_text = self._contextual_text_emb(images, prompts)

        # Caption branch
        if CFG.CAPTION_ON:
            ctx_cap = self._run_gcla("cap", q_text, fmap)
            ctx_bar = ctx_cap.mean(dim=1)
            prefix  = self.prefix_adapter(ctx_bar) if CFG.PREFIX_ON else torch.zeros(
                (q_text.size(0), 0, self.vlm.d_lm), device=q_text.device, dtype=q_text.dtype
            )
            loss_cap = self.vlm.caption_ce_with_prefix(images, prompts, answers, prefix)
        else:
            loss_cap = torch.tensor(0.0, device=q_text.device)

        # Gaze alignment
        if CFG.ALIGN_ON:
            ctx_gaze = self._run_gcla("gaze", q_text, fmap)
            ctx_gaze_bar = ctx_gaze.mean(dim=1)

            g = self.vlm.grid
            pred_gaze_down = F.interpolate(pred_gaze, size=(g, g), mode="bilinear", align_corners=False)
            pred_gaze_down = pred_gaze_down / pred_gaze_down.sum(dim=(1,2,3), keepdim=True).clamp_min(1e-8)

            B, C, H, W = fmap.shape
            F_flat = fmap.view(B, C, -1)
            w = pred_gaze_down.view(B, 1, -1)
            z_pred = torch.bmm(F_flat, w.transpose(1,2)).squeeze(-1)

            u_vis  = self.proj_vis(z_pred)
            u_txt  = self.proj_txt(ctx_gaze_bar)
            loss_align = info_nce(u_vis, u_txt, tau=CFG.ALIGN_TAU)
        else:
            loss_align = torch.tensor(0.0, device=q_text.device)

        wt = self.loss_w
        # if user has toggled a branch off, safety-zero its weight
        w_cap   = wt.get("cap", 1.0)   if CFG.CAPTION_ON else 0.0
        w_align = wt.get("align", 0.0) if CFG.ALIGN_ON   else 0.0

        loss_total = wt.get("gaze", 1.0) * loss_gaze + w_cap * loss_cap + w_align * loss_align

        if not self._dbg_once:
            print(f"[train] e{self.current_epoch} "
                  f"gaze={loss_gaze.item():.4f} align={loss_align.item():.4f} "
                  f"cap={loss_cap.item():.4f} "
                  f"kl_sharp={glogs['kl_sharp']:.4f} kl_blur={glogs['kl_blur']:.4f} gap={glogs['gap_pos']:.4f}")
            self._dbg_once = True

        logs = dict(
            loss_total=float(loss_total.item()),
            loss_gaze=float(loss_gaze.item()),
            loss_align=float(loss_align.item()),
            loss_cap=float(loss_cap.item()),
            kl_sharp=float(glogs["kl_sharp"]),
            kl_blur=float(glogs["kl_blur"]),
            gap_pos=float(glogs["gap_pos"]),
        )
        return loss_total, logs

    @torch.no_grad()
    def forward_eval(self, batch: Dict) -> Dict[str, float]:
        images  = batch["images"]
        prompts = batch["prompts"]
        answers = batch["answers"]
        gaze_gt = batch["gaze"].to(CFG.DEVICE)

        _, fmap = self.vlm.encode(images)
        pred_gaze = self.gaze_head(fmap)

        if CFG.GAZE_BLUR_GAP_ON:
            loss_gaze, glogs = gaze_loss_blur_gap(
                p_gt=gaze_gt, q_pred=pred_gaze,
                sigma=CFG.BG_SIGMA, lam=CFG.BG_LAMBDA, margin=CFG.BG_MARGIN
            )
        else:
            loss_gaze = gaze_loss_kl_only(gaze_gt, pred_gaze)
            glogs = {"kl_sharp": float("nan"), "kl_blur": float("nan"), "gap_pos": float("nan")}

        q_text = self._contextual_text_emb(images, prompts)

        if CFG.CAPTION_ON:
            ctx_cap = self._run_gcla("cap", q_text, fmap)
            ctx_bar = ctx_cap.mean(dim=1)
            prefix  = self.prefix_adapter(ctx_bar) if CFG.PREFIX_ON else torch.zeros(
                (q_text.size(0), 0, self.vlm.d_lm), device=q_text.device, dtype=q_text.dtype
            )
            loss_cap = self.vlm.caption_ce_with_prefix(images, prompts, answers, prefix)
        else:
            loss_cap = torch.tensor(0.0, device=q_text.device)

        if CFG.ALIGN_ON:
            ctx_gaze = self._run_gcla("gaze", q_text, fmap)
            ctx_gaze_bar = ctx_gaze.mean(dim=1)

            g = self.vlm.grid
            pred_gaze_down = F.interpolate(pred_gaze, size=(g, g), mode="bilinear", align_corners=False)
            pred_gaze_down = pred_gaze_down / pred_gaze_down.sum(dim=(1,2,3), keepdim=True).clamp_min(1e-8)

            B, C, H, W = fmap.shape
            F_flat = fmap.view(B, C, -1)
            w = pred_gaze_down.view(B, 1, -1)
            z_pred = torch.bmm(F_flat, w.transpose(1,2)).squeeze(-1)

            u_vis  = self.proj_vis(z_pred)
            u_txt  = self.proj_txt(ctx_gaze_bar)
            loss_align = info_nce(u_vis, u_txt, tau=CFG.ALIGN_TAU)
        else:
            loss_align = torch.tensor(0.0, device=q_text.device)

        wt = self.loss_w
        w_cap   = wt.get("cap", 1.0)   if CFG.CAPTION_ON else 0.0
        w_align = wt.get("align", 0.0) if CFG.ALIGN_ON   else 0.0
        loss_total = wt.get("gaze", 1.0) * loss_gaze + w_cap * loss_cap + w_align * loss_align

        return dict(
            loss_total=float(loss_total.item()),
            loss_gaze=float(loss_gaze.item()),
            loss_align=float(loss_align.item()),
            loss_cap=float(loss_cap.item()),
            kl_sharp=float(glogs["kl_sharp"]),
            kl_blur=float(glogs["kl_blur"]),
            gap_pos=float(glogs["gap_pos"]),
        )

    @torch.no_grad()
    def _contextual_text_emb(self, images: List[Image.Image], prompts: List[str]) -> torch.Tensor:
        msgs = []
        for q in prompts:
            q_clean = q.replace("<image>", "").strip()
            msgs.append([{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q_clean}]}])

        texts = self.vlm.processor.apply_chat_template(msgs, add_generation_prompt=False, tokenize=False)
        batch = self.vlm.processor(images=images, text=texts, return_tensors="pt", padding=True)
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                if k == "pixel_values":
                    batch[k] = v.to(self.vlm.device, dtype=self.vlm.model.dtype)
                else:
                    batch[k] = v.to(self.vlm.device)

        out = self.vlm.model(**batch, output_hidden_states=True, use_cache=False, return_dict=True)
        hs_mm = out.hidden_states[-1]
        input_ids = batch["input_ids"]
        attn_text = batch.get("attention_mask", None)
        if attn_text is None:
            attn_text = torch.ones_like(input_ids)
        L_text = input_ids.size(1)
        hs_text = hs_mm[:, -L_text:, :]

        am = attn_text.to(hs_text.device).to(hs_text.dtype)
        denom = am.sum(dim=1, keepdim=True).clamp_min(1.0)
        q_text = (hs_text * am.unsqueeze(-1)).sum(dim=1) / denom
        return q_text.to(torch.float32)


def make_optimizer(model: 'JointModel') -> torch.optim.Optimizer:
    def wd_groups(named_params):
        decay, no_decay = [], []
        for n, p in named_params:
            if p.requires_grad:
                (decay if p.ndim >= 2 else no_decay).append(p)
        return [
            {"params": decay, "weight_decay": 0.05},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    # language model params may be frozen if LoRA is off
    if CFG.LORA_ON:
        lora_named = list(model.vlm.model.language_model.named_parameters())
    else:
        lora_named = []  # nothing trainable in LM if LoRA is off

    heads_named = (
        list(model.gaze_head.named_parameters())
        + (list(model.gcla_cap.named_parameters()) if CFG.GCLA_MODE in ("dual", "shared") else [])
        + (list(model.gcla_gaze.named_parameters()) if CFG.GCLA_MODE == "dual" else [])
        + list(model.text2multi_cap.named_parameters())
        + list(model.text2multi_gaze.named_parameters())
        + list(model.proj_vis.named_parameters())
        + list(model.proj_txt.named_parameters())
        + list(model.prefix_adapter.named_parameters())
    )

    pg = []
    if lora_named:
        for g in wd_groups(lora_named):
            g["lr"] = CFG.LR_LORA
            pg.append(g)
    for g in wd_groups(heads_named):
        g["lr"] = CFG.LR_HEADS
        pg.append(g)

    print(f"[optim] lr_lora={CFG.LR_LORA if lora_named else 0.0} lr_heads={CFG.LR_HEADS}")
    opt = torch.optim.AdamW(pg, betas=(0.9, 0.95))
    return opt
