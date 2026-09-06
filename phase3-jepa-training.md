# JEPA-for-LOB Project — Phase 3: JEPA Model Design, Training & Ablation

**Status: planning document. Every architectural, numerical, and protocol decision below is final. Nothing is left for an implementing agent to infer, estimate, or improve upon. If something genuinely isn't covered here, stop and ask — do not extrapolate from what is.**

**This document assumes Phase 1 (six baselines) and Phase 2 (downstream-probe harness) are both complete and closed out.** Every empirical claim used to justify a decision below (which backbone, what "good" transfer looks like) is drawn from those two phases' actual results, not from the literature or from assumption.

---

## 0. What Phase 3 is, and what it deliberately is not

Phase 3 trains JEPA-style encoders on the same five-stock LOB data, evaluates them through the *exact same* Phase 2 harness the six baselines already went through, and runs one specific ablation that tests this project's actual novel claim. It is split into two sub-phases, deliberately sequenced from simpler to harder:

- **Phase 3a — JEPA-Temporal.** Masking along the time axis only, matching what generic time-series JEPA work (TS-JEPA, LaT-PFN) already does. This exists to de-risk the harder engineering — does the training loop even work, does the encoder avoid collapsing, does it integrate cleanly with the Phase 2 harness — before adding the untested part.
- **Phase 3b — JEPA-SpatioTemporal.** Identical to 3a in every other respect, except masking also operates along the LOB's field-type axis, not just time. **This is the actual contribution this project is built around** — nothing in the literature review from the start of this project found a JEPA variant that masks along both axes for LOB data. 3a exists so that 3b's result is interpretable as "does the spatial masking help," not "does JEPA work at all."

Both are built on the same reused backbone, same data pipeline, same collapse-prevention mechanism, same training protocol — the *only* thing that differs between 3a and 3b is the masking function. This is deliberate: it's the same discipline as Phase 1's shared-decoder principle, applied here to isolate the one variable that's actually being tested.

**What Phase 3 does not include:** hyperparameter search across many JEPA variants, additional backbone families beyond the one justified in Section 1, or the 5×4 full transfer matrix (still deferred per Phase 2's own Section 4.3). Those are legitimate future work, not blocking this phase.

---

## 1. Backbone architecture — decision and justification

**Decision: the JEPA context and target encoders reuse SimLOB's `FCN1 → Transformer stack` exactly, with SimLOB's own `FCN2` reduction stage removed.**

**Why SimLOB, empirically, not by assumption:** across Phase 1's reconstruction results, SimLOB had the lowest test MSE on 4 of 5 stocks and the best full-mean MSE overall. Across Phase 2, SimLOB's frozen encoder produced the best or near-best trend-prediction Macro-F1 in most stocks. This project has direct, verified evidence that SimLOB's encoder architecture extracts the most useful representation of this data among the six tested — that is the entire basis for this choice, not a preference from the literature.

**Why remove `FCN2` specifically:** SimLOB's `FCN2` (Section 3.5 of the Phase 1 document — `reduce_proj` then a 3-layer MLP down to a single 256-dim vector) exists to *pool* a per-timestep sequence into one global latent, which is what Phase 1's reconstruction task and Phase 2's frozen-probe tasks need. **JEPA needs the opposite** — a per-timestep (or per-patch) embedding at every position, so that a subset of positions can be masked and predicted individually. Reusing `FCN1` (the 40→256 per-timestep projection) and the 2-layer Transformer stack gives exactly a `[B, 100, 256]` sequence output, which is what JEPA operates on. This is not a new architecture invented for this phase — it is SimLOB's own already-verified encoder body, used up to the point where SimLOB itself stops needing per-position information.

```python
# Reused verbatim from the Phase 1 SimLOBEncoder (Section 3.5 of the baseline master doc),
# with reduce_proj/fcn2 removed. Output is per-timestep, not pooled.
class JEPABackbone(nn.Module):
    def __init__(self, n_features=40, d_model=256, nhead=8, num_layers=2, dim_feedforward=512):
        super().__init__()
        self.fcn1 = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):  # x: [B, 100, 40] (already masked, if this is the context encoder)
        h = self.fcn1(x)          # [B, 100, 256]
        h = self.transformer(h)   # [B, 100, 256]
        return h
```

**Parameter count check (mandatory, do not skip):** instantiate this and print `sum(p.numel() for p in m.parameters())` before proceeding — it should be close to SimLOB's own Phase 1 encoder count (~5.8M, per the empirical check run when Phase 1's code was reviewed) minus whatever `FCN2`/`reduce_proj` contributed. If it's wildly different, something was copied wrong.

---

## 2. Full JEPA architecture

### 2.1 Context encoder and target encoder

Two instances of `JEPABackbone` from Section 1:
- **Context encoder** — receives the *masked* input, trained via backpropagation.
- **Target encoder** — receives the *full, unmasked* input, updated only via exponential moving average (EMA) of the context encoder's weights (Section 4). It never receives a gradient directly.

```python
context_encoder = JEPABackbone()
target_encoder  = JEPABackbone()
target_encoder.load_state_dict(context_encoder.state_dict())  # identical init
for p in target_encoder.parameters():
    p.requires_grad = False   # never trained by backprop, only by EMA
```

### 2.2 Masking mechanism — deliberate simplification, stated explicitly

**Decision: masked positions are replaced with a learnable mask-token vector at the input level, keeping sequence length fixed at 100 throughout.** This is a deliberate departure from "pure" I-JEPA, which removes masked patches from the context encoder's input entirely (variable-length sequences). We use the MAE-style fixed-length approach instead — masked timesteps' 40 input features are replaced by a single learnable `[40]`-dim mask-token vector (broadcast to whichever positions are masked) before the context encoder's `fcn1` layer.

**Why this simplification, stated plainly:** variable-length sequence handling (padding, attention masks, unmasking-index bookkeeping) is a meaningfully higher implementation-risk pattern than fixed-length masking, and this project's implementing agents have a demonstrated track record of introducing subtle indexing bugs in exactly this kind of bookkeeping (the `m_future_shifted` off-by-`k` bug from Phase 2 is the direct precedent). Fixed-length masking with a mask token is a well-established, lower-risk alternative used in MAE and several practical JEPA implementations, and it preserves the actual thing being tested (predicting target representations at masked positions from context at visible ones) without the variable-length complexity. This is a documented, justified engineering choice — not a shortcut taken silently.

```python
mask_token = nn.Parameter(torch.zeros(1, 1, 40))
nn.init.trunc_normal_(mask_token, std=0.02)

def apply_mask(x, mask):  # x: [B, 100, 40], mask: [B, 100] bool, True = masked
    x_masked = x.clone()
    x_masked[mask] = mask_token.squeeze()  # broadcast mask token to every masked position
    return x_masked
```

### 2.3 Predictor

A separate, smaller Transformer — deliberately narrower than the encoder, matching the established pattern in the JEPA literature (I-JEPA's predictor is narrower than its encoder) since the predictor's job (map context representations + positional info to target representations) is simpler than the encoder's job (build a rich representation from raw input).

```python
class Predictor(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4, dim_feedforward=512, seq_len=100):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, context_output):  # [B, 100, 256], already includes context encoder's
                                          # output at visible positions and its own (weaker)
                                          # output at masked positions -- see Section 3
        h = context_output + self.pos_embedding
        h = self.transformer(h)
        return self.output_proj(h)  # [B, 100, 256], prediction of target encoder's output
```

**Why 4 layers for the predictor versus 2 for the encoder:** matches the established convention (predictor deeper is fine, predictor *narrower* in `d_model` is the usual I-JEPA choice, but we keep `d_model=256` identical to the encoder here specifically to avoid needing an extra projection layer between context encoder output and predictor input — one less place for a dimension-mismatch bug, consistent with this project's preference for minimizing surface area for silent errors over squeezing out marginal efficiency).

---

## 3. Training step — exact data flow

This is the part most likely to be implemented subtly wrong, so it's given as literal, ready-to-use code, not prose description.

```python
def jepa_training_step(x, mask_fn, context_encoder, target_encoder, predictor, mask_token):
    """
    x: [B, 100, 40], a raw (unmasked) window from the dataloader -- same windows,
       same split, same session logic as every prior phase (Section 5).
    mask_fn: either temporal_mask() (Phase 3a) or spatiotemporal_mask() (Phase 3b) -- Section 6.
    """
    B = x.shape[0]
    mask = mask_fn(B)  # [B, 100] bool, True = masked position (Section 6 defines exact shapes)

    # 1. Context encoder sees the masked input.
    x_masked = apply_mask(x, mask)
    context_output = context_encoder(x_masked)          # [B, 100, 256]

    # 2. Target encoder sees the FULL, unmasked input, no gradient.
    with torch.no_grad():
        target_output = target_encoder(x)                # [B, 100, 256]
        target_output = target_output.detach()

    # 3. Predictor takes the context encoder's output (informative at visible positions,
    #    only mask-token-derived at masked ones) and predicts the target encoder's output
    #    everywhere -- but the loss is only computed at masked positions (Section 3.1).
    predicted_output = predictor(context_output)          # [B, 100, 256]

    # 4. Loss only at masked positions.
    loss_pred = prediction_loss(predicted_output, target_output, mask)   # Section 3.1
    loss_collapse = collapse_prevention_loss(context_output)              # Section 3.2

    total_loss = loss_pred + LAMBDA_COLLAPSE * loss_collapse
    return total_loss, loss_pred, loss_collapse
```

### 3.1 Prediction loss — masked positions only

Same principle as Phase 2's imputation loss (Section 3.3 of that document) — restrict to masked positions, for the same reason: computing loss over visible positions too would let the model trivially minimize loss by copying context-encoder output to predictor output where nothing needed predicting.

```python
def prediction_loss(pred, target, mask):  # mask: [B, 100] bool, True = masked
    diff = (pred - target) ** 2
    masked_diff = diff[mask.unsqueeze(-1).expand_as(diff)]
    return masked_diff.mean()
```

### 3.2 Collapse-prevention loss — VICReg-style, not SIGReg, and here is why

**Decision: use VICReg's variance + covariance regularization (Bardes, Ponce & LeCun, 2022), not LeJEPA's SIGReg.** This needs to be stated honestly: SIGReg was the mechanism referenced throughout this project's earlier research (Section 1.1 of the Phase 1 document, and Fin-JEPA's own use of it), and it is likely the better-performing choice long-term. But this document holds every other number to the standard of "verified against a primary source before being written down as a mandate" — and SIGReg's exact sketched-projection formula was not independently re-derived or verified with that same rigor before this document was written. Writing an unverified formula into a document whose entire premise is "no ambiguity, no guessing" would violate the standard this project has held every implementing agent to. **VICReg's formula is precisely specified below, from memory that is genuinely confident, not approximate** — use it now; revisit SIGReg later as a documented follow-up once its formula is pulled directly from the LeJEPA paper or code and verified the same way DeepLOB's architecture was verified in Phase 1.

```python
def collapse_prevention_loss(z, gamma=1.0, eps=1e-4):
    """
    z: [B, 100, 256] -- context encoder output, flattened to [B*100, 256] for this computation.
    Variance term: penalize any embedding dimension whose std (across the batch) falls
    below gamma -- this is what actually prevents collapse to a constant vector.
    Covariance term: decorrelate different embedding dimensions from each other.
    """
    z = z.reshape(-1, z.shape[-1])              # [B*100, 256]
    z = z - z.mean(dim=0)

    std_z = torch.sqrt(z.var(dim=0) + eps)
    loss_var = torch.mean(F.relu(gamma - std_z))

    N, D = z.shape
    cov_z = (z.T @ z) / (N - 1)                  # [256, 256]
    off_diagonal = cov_z.flatten()[:-1].view(D - 1, D + 1)[:, 1:].flatten()  # off-diag elements
    loss_cov = off_diagonal.pow(2).sum() / D

    return loss_var + loss_cov  # equal weighting between the two sub-terms
```

`LAMBDA_COLLAPSE = 1.0` (i.e., prediction loss and collapse-prevention loss are weighted equally in the total loss). This matches VICReg's own paper's practice of using comparable-magnitude coefficients for its variance/covariance/invariance terms rather than a large imbalance; do not tune this without explicitly noting the change and why.

---

## 4. EMA update for the target encoder

Standard momentum-encoder schedule, matching I-JEPA's own choice exactly (not reinvented): momentum starts at 0.996 and anneals linearly to 1.0 over the full course of training.

```python
def update_target_encoder(context_encoder, target_encoder, momentum):
    with torch.no_grad():
        for p_context, p_target in zip(context_encoder.parameters(), target_encoder.parameters()):
            p_target.data.mul_(momentum).add_(p_context.data, alpha=1 - momentum)

def get_momentum(step, total_steps, start=0.996, end=1.0):
    return start + (end - start) * (step / total_steps)
```

Call `update_target_encoder(...)` once per training step, immediately after the optimizer step on `context_encoder` and `predictor` (never on `target_encoder` directly via the optimizer — it has no gradient, per Section 2.1).

---

## 5. Data pipeline — 100% reused, zero new logic

**This section has no new code, on purpose.** Every function needed already exists and is frozen from Phase 1/2:

- `split_by_date`, `detect_sessions`, `get_window_indices` — imported directly from `common.py`, exactly as Phase 2's corrected `downstream_common.py` does. **Do not reimplement these a third time.** This is the same rule as Phase 2's Section 0b, and it applies with the same force here.
- Window shape, stride, and session-awareness: identical to Phase 1 — `[100, 40]` windows, stride 1 within a session, never crossing a session or split boundary.
- Seed: 42, set via the same `set_seed()` function already in `downstream_common.py` — import it, don't recreate it.
- Batch size: 256, same as every prior phase, for the same reason (established, working, comparable).
- `num_workers=2`, same Colab-CPU-count reasoning as Phase 1's Section 2.8.

**The only new data-pipeline decision in this phase is which stocks to train on.** Train the JEPA encoder on **all five stocks jointly** (not one model per stock, unlike Phase 1's baselines) — the whole reason JEPA is being tested here is its capacity to learn transferable, cross-stock structure, so training it on a single stock and then testing transfer would be a weaker, less representative test of that specific claim than training on the general distribution across stocks directly. Concretely: pool the training-split windows from all five stocks into one dataset for JEPA pretraining. This is a deliberate, one-time decision — the reasoning is written here so it doesn't need to be re-derived or second-guessed later.

---

## 6. Masking functions — the actual experimental variable

### 6.1 Phase 3a — Temporal-only masking

Multi-block masking along the time axis, following I-JEPA's own block-masking convention (several blocks rather than one, since a single contiguous block is easier to "cheat" on by interpolating from both edges).

```python
def temporal_mask(batch_size, seq_len=100, n_blocks=4, block_size_range=(10, 20)):
    """Returns [B, 100] bool tensor, True = masked. Each sample gets n_blocks
    non-overlapping temporal blocks masked, block lengths drawn uniformly from block_size_range."""
    masks = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    for b in range(batch_size):
        occupied = torch.zeros(seq_len, dtype=torch.bool)
        blocks_placed = 0
        attempts = 0
        while blocks_placed < n_blocks and attempts < 50:
            attempts += 1
            block_len = torch.randint(block_size_range[0], block_size_range[1] + 1, (1,)).item()
            start = torch.randint(0, seq_len - block_len + 1, (1,)).item()
            span = slice(start, start + block_len)
            if occupied[span].any():
                continue   # overlaps an existing block, retry
            occupied[span] = True
            blocks_placed += 1
        masks[b] = occupied
    return masks
```

`attempts < 50` is a safety valve, not a tunable — with 4 blocks of length 10-20 in a 100-length sequence (40-80 positions out of 100), rejection sampling converges quickly; this cap exists only to guarantee termination, not to be hit in normal operation. **If this cap is ever actually hit in practice** (check by logging `attempts` and `blocks_placed` at least once during initial testing), that's a sign the block-size range needs revisiting — flag it rather than silently accepting fewer than 4 blocks.

### 6.2 Phase 3b — Spatio-temporal masking (the actual contribution)

Identical temporal block-masking as 3a, **plus** partial feature-group masking on a fraction of the timesteps that survived as "visible." Field groups use the LOB schema's already-established, unambiguous boundaries (Section 2.2 of the Phase 1 document — no new reordering, no risk of repeating the FI-2010 reordering mistake): `BidPrice` = columns 0-9, `AskPrice` = columns 10-19, `BidVolume` = columns 20-29, `AskVolume` = columns 30-39.

```python
FIELD_GROUPS = {
    'BidPrice': slice(0, 10),
    'AskPrice': slice(10, 20),
    'BidVolume': slice(20, 30),
    'AskVolume': slice(30, 40),
}

def spatiotemporal_mask(x, batch_size, seq_len=100, n_blocks=4, block_size_range=(10, 20),
                          field_mask_prob=0.3):
    """Returns (temporal_mask, field_mask_info). temporal_mask is identical to Section 6.1's
    output. field_mask_info additionally zeroes out one randomly chosen field group, for a
    field_mask_prob fraction of the temporally-VISIBLE positions, applied to the input tensor
    directly (this is feature-level corruption, not a full-timestep mask, so it modifies x
    in place rather than returning a second boolean mask of the same shape)."""
    temporal_masked = temporal_mask(batch_size, seq_len, n_blocks, block_size_range)
    x_out = x.clone()
    field_names = list(FIELD_GROUPS.keys())
    for b in range(batch_size):
        visible_positions = (~temporal_masked[b]).nonzero(as_tuple=True)[0]
        n_to_corrupt = int(len(visible_positions) * field_mask_prob)
        if n_to_corrupt == 0:
            continue
        chosen_positions = visible_positions[torch.randperm(len(visible_positions))[:n_to_corrupt]]
        for pos in chosen_positions:
            field = field_names[torch.randint(0, len(field_names), (1,)).item()]
            x_out[b, pos, FIELD_GROUPS[field]] = 0.0   # zero out that field group at that timestep
    return temporal_masked, x_out
```

**Important distinction to keep straight when implementing the training step:** in Phase 3b, `apply_mask` (Section 2.2) still only replaces *temporally masked* positions with the learnable mask token. The field-group corruption is a *second, separate* input-level operation applied to the temporally-visible positions, using zeros (not the mask token) — this is a deliberate distinction: full mask-token replacement signals "this entire position is unknown," while zeroing one field group at an otherwise-visible timestep signals "this position is known except for this one measurement," and the model should be able to learn to tell these two situations apart from the input alone. Do not conflate the two into a single mechanism.

### 6.3 Ablation protocol

Train two encoders, identical in every respect (backbone, predictor, collapse loss, EMA schedule, data, epochs, seed) except the masking function — 3a uses `temporal_mask`, 3b uses `spatiotemporal_mask`. Both get evaluated through the *identical* Phase 2 harness (Section 7). The comparison between 3a and 3b's downstream numbers is the actual experimental result this whole project has been building toward. Report both, always side by side, never one without the other.

---

## 7. Evaluation — reuses Phase 2's harness with zero new code

This is the payoff for building Phase 2 as a model-agnostic harness. Once a JEPA context encoder is trained (3a or 3b), it is frozen and evaluated exactly like the six Phase 1 baselines were:

1. **Checkpoint saving convention**: identical to Phase 1/2 — `save_top_k=1`, `save_last=True`, monitor the total training loss (Section 3), and use the *exact same* `select_optimal_checkpoint()` function from `downstream_common.py` to load it later. Do not write a new checkpoint-selection function for JEPA — this is precisely the function that was audited and fixed across two rounds of review; reusing it is not optional.
2. **One architectural adapter needed, and only one**: Phase 2's `load_frozen_encoder` and the three probes (`TrendHead`, imputation, transfer) all expect an encoder that outputs a single `[B, 256]` vector per window, since that's what all six Phase 1 baselines produce. JEPA's context encoder outputs `[B, 100, 256]` (per-timestep). **Decision: mean-pool over the time dimension** (`context_output.mean(dim=1)`) to produce a `[B, 256]` vector for evaluation purposes only — this pooling is not part of the JEPA model itself, it is a thin adapter applied at evaluation time so the *exact* same `TrendHead`, imputation loop, and transfer protocol can run unmodified. Write this adapter as a small wrapper class, not as a change to `downstream_common.py`:

```python
class JEPAEncoderForEval(nn.Module):
    """Wraps a trained JEPA context encoder so it presents the same [B,256] output
    interface Phase 2's harness expects from every Phase 1 baseline encoder.
    Does not modify downstream_common.py -- this wrapper lives in the new Phase 3 module."""
    def __init__(self, jepa_backbone):
        super().__init__()
        self.backbone = jepa_backbone

    def forward(self, x):  # x: [B, 100, 40], UNMASKED -- evaluation always uses the full window
        h = self.backbone(x)          # [B, 100, 256]
        return h.mean(dim=1)           # [B, 256] -- matches every Phase 1/2 encoder's output shape
```
3. **Everything downstream of that wrapper is unchanged**: `select_optimal_checkpoint`, latent caching, `TrendHead`, the imputation probe, and the transfer protocol all run exactly as specified in the Phase 2 document, with `JEPAEncoderForEval`-wrapped models substituted in wherever a Phase 1 encoder used to go. This is not a suggestion to keep it similar — it must be the literal same functions, imported the same way, for the comparison to mean anything.

---

## 8. Training protocol

- **Optimizer**: Adam, lr=1e-4 — same as every prior phase, for the same reason (established, consistent, avoids introducing a new untested hyperparameter regime).
- **Epochs**: 100, matching Phase 1's convention. Given JEPA trains on the pooled five-stock dataset (Section 5) rather than one stock at a time, this is a materially larger amount of data per epoch than any single Phase 1 baseline saw — budget accordingly (Section 9).
- **Batch size**: 256, `num_workers=2` — unchanged from every prior phase.
- **Checkpointing**: every epoch, `save_top_k=1` + `save_last=True`, monitoring total training loss (Section 3) on a held-out validation slice of the pooled dataset (use the same per-stock val-split windows, pooled the same way as train).
- **Seed**: 42, via the existing `set_seed()`, called once at the top of each of the two (3a, 3b) training notebooks — and per Phase 2's already-fixed lesson, if training is ever resumed after an interruption mid-epoch, that's a single continuous `trainer.fit(..., ckpt_path=...)` resume (Lightning's own resume, as used throughout Phase 1), not a "skip completed pairs and reseed" loop like Phase 2's per-pair probes — so Phase 2's specific reseeding fix does not apply here; this is a different resumption shape entirely (one long-running training job, not many independent short ones), and should use Phase 1's `common.py` resume pattern instead.

---

## 9. Coding rules, folder structure, and frozen artifacts

Same discipline as every phase before this one, extended to cover Phase 1 and Phase 2's outputs both:

```
JEPA_LOB/baselines/
├── common.py                     <- Phase 1. Read-only.
├── checkpoints/                  <- Phase 1. Read-only.
├── training_logs/                <- Phase 1. Read-only.
├── LSTM.ipynb ... TransLOB.ipynb  <- Phase 1. Do not re-run.
├── downstream_common.py          <- Phase 2. Read-only.
├── latents/, downstream_checkpoints/, downstream_results/  <- Phase 2. Read-only.
├── TrendPrediction.ipynb, Imputation.ipynb, Transfer.ipynb  <- Phase 2. Do not re-run.
│
├── jepa_common.py                <- NEW, Phase 3 only: JEPABackbone, Predictor, both masking
│                                      functions, collapse_prevention_loss, EMA update, and
│                                      JEPAEncoderForEval.
├── jepa_checkpoints/{3a,3b}/     <- NEW, Phase 3 only
├── jepa_training_logs/{3a,3b}/   <- NEW, Phase 3 only
├── JEPA_Temporal_Train.ipynb     <- NEW, Phase 3a training
├── JEPA_SpatioTemporal_Train.ipynb <- NEW, Phase 3b training
└── JEPA_Evaluate.ipynb           <- NEW: runs both 3a and 3b through the Phase 2 harness
                                      (via the wrapper in Section 7), produces the final
                                      side-by-side comparison table against all 6 baselines.
```

**If anything about this phase seems to require changing a Phase 1 or Phase 2 file, stop and say so — same rule as every phase document before this one.** The only exception, already anticipated: if `downstream_common.py`'s probes turn out to assume something about their input encoder beyond "produces `[B,256]`" that JEPA's wrapper can't satisfy, that's worth surfacing as a real question before working around it silently.

**Evidence standard**: unchanged from Phase 2's Section 5 — every claim of "trained successfully," "collapse avoided," or "matches spec" needs the actual raw output attached (loss curves, a printed check that `context_output`'s per-dimension std is meaningfully above zero — a cheap, direct collapse check worth running and reporting explicitly, not just inferring collapse-avoidance from the loss curve looking reasonable).

---

## 10. Budget and compute planning

Two training runs (3a, 3b), each on the pooled five-stock dataset — roughly 5x the per-epoch data volume of any single Phase 1 baseline run. Using Phase 1's own empirically-observed T4 timings as the reference point (a single-stock baseline of comparable parameter count trained in a few hours per 100 epochs), budget **on the order of a full day of T4 time per JEPA variant**, not a few hours — this is a real, larger compute commitment than any single Phase 1 or Phase 2 run, and should be explicitly weighed against remaining account/session budget before starting both variants in parallel. Given the project's established multi-account parallel-training pattern, running 3a and 3b simultaneously on two separate accounts is the natural choice, rather than sequentially on one.

**Do not reduce epochs or dataset pooling to fit a smaller budget without flagging it here first** — unlike Phase 1's baselines, where each was independently comparable at 100 epochs on its own stock, 3a and 3b's comparison is only meaningful if both get the identical budget. An asymmetric cut to one but not the other would confound the ablation.

---

## 11. What is explicitly deferred, not decided by omission

- **SIGReg as a collapse-prevention upgrade over VICReg** (Section 3.2) — real candidate, deliberately deferred until its exact formula is independently verified against the LeJEPA paper/code with the same rigor as everything else in this document.
- **A third JEPA variant using price-level (10-level) patches rather than the simpler 4-field-group patches used in 3b** — the field-group design (Section 6.2) was chosen specifically to avoid a new, unverified column-reordering step; a level-based version remains a legitimate follow-up once 3a/3b results are in.
- **The full 5×4 transfer matrix** — still deferred from Phase 2, applies identically here once JEPA encoders exist.
- **TimesNet/iTransformer as 7th/8th baselines** — still deferred from the original Phase 1 scoping decision, revisit once Phase 3's results and remaining budget are both known.
