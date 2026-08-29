# JEPA-for-LOB Project — Phase 2: Downstream-Probe Evaluation Harness

**Status: planning document. Nothing in this document has been implemented or run yet — this defines the spec before any code is written, same discipline as Phase 1's master document.**

**Resolved:** the earlier "sz300147 checkpoint question" turned out to be a real bug, not a false alarm — see Section 0a below. It's the single most relevant lesson from Phase 1 for everything in this document, since Phase 2 loads the exact same checkpoints again.

---

## 0a. Lessons from Phase 1 — read this before writing any Phase 2 code

Phase 1 is done, but it did not go cleanly, and every mistake below is a specific, real incident from this project, not a hypothetical. Each one has a rule attached. Follow the rule; do not re-derive the reasoning from scratch and land somewhere else.

1. **The checkpoint-selection bug.** `ModelCheckpoint(save_top_k=1, ...)` loses its internal "best score so far" state across a Colab session restart. A resumed session can save a *worse* epoch as `best-v1.ckpt`/`best-v2.ckpt` sitting right next to an earlier, genuinely better `best.ckpt` — and naive loading (by filename pattern, or by latest modification time) silently picks the worse one. This was found in **6 of 30 Phase 1 cells** despite an earlier agent explicitly asserting the results were "100% verified." **Rule for Phase 2:** every place this document loads a checkpoint (Section 1 and 4), it must scan *every* `best*.ckpt` file in that directory, read the actual recorded validation score from each one's metadata, and load whichever file has the true lowest value — never infer this from a filename, a version suffix, or a timestamp. Section 1a below gives the exact required code.
2. **Fabricated or unverified claims, stated with full confidence.** Across Phase 1, we had: a cited code snippet that did not exist anywhere in the file it was attributed to; a claimed data file (`pv_100w.pt`) that does not exist in the public repository; a "fix" (a decoder signature patch) that was asserted to work but, when actually executed, crashed one line later than claimed. In every case, the claim was stated as settled fact, not as something uncertain. **Rule for Phase 2:** any claim that something "works," "is verified," or "is fixed" must be accompanied by the actual raw output that demonstrates it — the literal printed shapes, the literal error-free execution log, the literal checkpoint metadata dump. A narrative description of what should have happened is not evidence. If you cannot produce the raw output, say so explicitly instead of asserting success.
3. **Silent architectural drift from spec.** The reason Phase 1's baseline notebooks turned out reliable on inspection is that every encoder was checked character-for-character against the master document, and then actually executed with dummy tensors before being trusted. **Rule for Phase 2:** every piece of new code in this document (label computation, masking, head architectures, transfer protocol) gets the same treatment — implemented exactly as specified below, then run once on dummy or small real data to confirm shapes and behavior, before being trusted at scale.
4. **Leakage introduced by convenience shortcuts.** The original LOBench pipeline's `random_split` on overlapping windows, and this project's own early mistake of computing a threshold or statistic across an entire dataset instead of the train split only, are the same category of error: using information from data a model should not have seen yet. **Rule for Phase 2:** anywhere this document says "compute X from the train split only" (Section 2.1's threshold, most importantly), that is not a suggestion — computing it from the full dataset, even by accident, silently invalidates every downstream number built on it.
5. **Device-mismatch and off-by-default bugs that only surface when code actually runs.** Cell 9's original evaluation loop crashed the first time it was actually executed (GPU model, CPU batch), despite looking correct on read-through. **Rule for Phase 2:** encoders must be moved to the same device as the cached-latent computation explicitly, `.eval()` and `torch.no_grad()` must be set explicitly when caching latents (Section 1), and this should be confirmed by actually running it, not by inspection alone.

## 0b. Phase 1 artifacts are frozen — do not touch them

This needs to be said as plainly as possible: **Phase 2 code must never modify, overwrite, delete, or retrain anything belonging to Phase 1.** Specifically:

- **`common.py` is read-only from Phase 2's perspective.** If Phase 2 needs something from it (an encoder class, `SharedDecoder`), import it or copy the class definition into the new Phase 2 module — do not edit `common.py` itself, even for something that looks like a small improvement.
- **Every file under `checkpoints/` is read-only.** Phase 2 only ever opens these files to read weights and metadata. Never write to, rename, move, or delete anything in this directory, including the now-identified "wrong" checkpoint files from the versioning bug (Lesson 1 above) — leave them exactly where they are; the corrected selection logic reads around them, it doesn't need them removed.
- **Do not re-run any Phase 1 training notebook** as part of Phase 2 work, for any reason, including "just to double check." If a Phase 1 result is in question, that's a Phase 1 conversation, resolved by inspecting existing checkpoints and logs — not by retraining.
- **Phase 2 gets its own folder and its own files, sitting alongside Phase 1's, never inside it or overwriting it:**
```
JEPA_LOB/baselines/
├── common.py                    <- Phase 1. Read-only. Do not edit.
├── checkpoints/                 <- Phase 1. Read-only. Do not touch.
├── training_logs/               <- Phase 1. Read-only.
├── LSTM.ipynb ... TransLOB.ipynb <- Phase 1. Do not re-run.
├── downstream_common.py         <- NEW, Phase 2 only
├── latents/                     <- NEW, Phase 2 only (Section 1)
├── downstream_checkpoints/      <- NEW, Phase 2 only (Section 5)
├── TrendPrediction.ipynb        <- NEW, Phase 2 only
├── Imputation.ipynb             <- NEW, Phase 2 only
└── Transfer.ipynb               <- NEW, Phase 2 only
```
- If anything about this phase seems to require changing a Phase 1 file, **stop and say so explicitly rather than making the edit** — that's a decision for this conversation, not something to resolve unilaterally.

---



## 0. Why this phase exists, and how it relates to Phase 1

Phase 1 (the six baseline notebooks, currently running) answers "can each encoder architecture learn to reconstruct a LOB window." That's necessary but not what this project's actual thesis rests on. Go back to the original research plan: the entire "why JEPA could plausibly work here" argument was built on LOBench's own Table 6 finding — a frozen, transferred encoder beating end-to-end-trained models on cross-stock generalization. That is the mechanism being tested, not reconstruction MSE. Phase 1 produces the trained encoders; Phase 2 is where the actual claim gets tested.

**Important distinction to hold onto throughout this document:** Part 1.5 of the Phase 1 master document explains why we don't cite LOBench's *published numbers* as ground truth (non-reproducible split, mischaracterized architecture, missing configs, our own 56.6% miss). That does **not** disqualify LOBench's *task definitions and protocols* (how trend labels are defined, how transfer is measured) from being reused as reasonable, citable methodology — those are design choices, not results, and are not tainted by the code/data problems we found. This document adopts several such protocol choices explicitly, and says so at each point, so it's always clear whether something is "a number we don't trust" (never used) versus "a task convention we're choosing to adopt" (fine to use, cited as methodology not as a result).

---

## 1. Universal downstream-evaluation principle

### 1a. Prerequisites — what must already exist, and what does not need to be rerun

**Phase 2 never retrains or reruns any Phase 1 encoder.** It only loads existing checkpoints (via the selection function below) and runs frozen forward passes. Nothing in Phase 1 needs to be redone because of anything in this document.

Checkpoint requirements differ by task — this affects sequencing, not just completeness:
- **Trend prediction (Section 2) and imputation (Section 3)** are same-stock tasks: each needs that specific (model, stock) pair's own trained encoder. Full coverage requires all 30 (6 models × 5 stocks) checkpoints, but each can be evaluated as soon as its own checkpoint exists — no need to wait for every model/stock combination to finish.
- **Transfer (Section 4)** only needs the **sz000001** checkpoint per model — 6 checkpoints, not 30. This is the cheapest prerequisite of the three, and the headline metric — a model's transfer evaluation can start the moment its sz000001 run finishes, in parallel with that model's other four stocks still training.



**All three probes use a fully frozen encoder.** No gradient flows into any encoder parameter during Phase 2. Only a small, task-specific head is trained on top. This is deliberate, not a simplification for convenience: it's the direct test of the actual claim (representation quality, not "can we still improve the encoder with more training on this task"). Fine-tuning the encoder is explicitly out of scope for this phase — if we want that as an ablation later, it gets its own clearly-separated experiment, never mixed into the primary numbers.

**Precompute and cache latents once — do not re-run the encoder for every downstream experiment.** Since the encoder is frozen, its output for a given window never changes. For each (model, stock) pair from Phase 1:
1. Load the trained encoder from its Phase 1 checkpoint — **using the exact selection function below, not `checkpoints/{model_name}/{stock}/best.ckpt` directly.** Per Lesson 1 (Section 0a), a fixed filename is not safe to assume is the true best checkpoint.

```python
import glob, torch

def select_optimal_checkpoint(ckpt_dir, metric_name='val_mse'):
    """Scan every best*.ckpt in ckpt_dir and return the path with the true lowest
    recorded validation score. Never infer this from filename, version suffix,
    or modification time -- read the actual stored metric from each file."""
    candidates = glob.glob(f"{ckpt_dir}/best*.ckpt")
    assert candidates, f"No checkpoint files found in {ckpt_dir}"
    scored = []
    for path in candidates:
        ckpt = torch.load(path, map_location='cpu')
        # Lightning stores the monitored metric under callback_metrics or
        # ModelCheckpoint's own state -- confirm the exact key by printing
        # ckpt.keys() / ckpt['callbacks'] once before trusting this in bulk.
        score = ckpt['callbacks'][next(k for k in ckpt['callbacks'] if 'ModelCheckpoint' in k)]['best_model_score']
        scored.append((path, float(score), ckpt.get('epoch')))
    scored.sort(key=lambda x: x[1])  # lowest score wins, regardless of filename
    best_path, best_score, best_epoch = scored[0]
    print(f"Selected {best_path} (epoch {best_epoch}, {metric_name}={best_score:.6f}) "
          f"from {len(candidates)} candidates in {ckpt_dir}")
    for path, score, epoch in scored[1:]:
        print(f"  (rejected: {path}, epoch {epoch}, {metric_name}={score:.6f})")
    return best_path
```
Print every rejected candidate alongside the selected one, every time — this is exactly the kind of output that caught the Phase 1 bug in the first place, and it costs nothing to keep printing it. Before using this at scale across all 30 directories, run it once on a single directory known to have multiple checkpoint files and manually confirm the printed score matches what you'd expect — do not trust the metadata key path (`ckpt['callbacks'][...]['best_model_score']`) blindly, since the exact structure can vary by Lightning version; print `ckpt.keys()` and `ckpt['callbacks']` directly if the above doesn't work on the first try, and adjust.
2. Run one forward pass over every window in the dataset (train + val + test), in `eval()` mode, `torch.no_grad()`.
3. Save the resulting `[N, 256]` latent array to disk (e.g., `latents/{model_name}/{stock}/{split}.npy`), alongside the corresponding labels needed for each probe (Section 2–4).

This decouples the one-time, GPU-bound encoding pass from the cheap, fast head-training that follows — all three downstream probes can then be trained and iterated on using only these small cached arrays, on CPU if needed, without touching the encoder or a T4 again. Given 6 models × 5 stocks × ~3 splits, this is 90 small `.npy` files — trivial storage compared to Phase 1's checkpoints.

---

## 2. Task 1 — Trend Prediction

### 2.1 Label definition (adopted convention, not a cited result)

We adopt the standard FI-2010-lineage trend-labeling convention (also used by LOBench, DeepLOB, and most of the cited literature), adapted to our own Z-score-normalized schema since we don't have — and can't reconstruct — LOBench's original normalization statistics:

- Mid-price at time `t`: `mid(t) = (BidPrice1(t) + AskPrice1(t)) / 2`, using the already-normalized columns from Section 2.2 of the Phase 1 document.
- Horizon: `k = 5` (5 ticks, i.e. 5 rows in our 3-second-sampled data — matches the "5-tick horizon" terminology used throughout the cited literature).
- Past average: `m_past(t) = mean(mid(t-k+1), ..., mid(t))`.
- Future average: `m_future(t) = mean(mid(t+1), ..., mid(t+k))`.
- Raw signal: `l(t) = m_future(t) - m_past(t)`. **We use the raw difference, not a percentage change** — percentage change is numerically unstable on Z-scored data (denominators can be near zero or negative), unlike on raw prices where the original literature's percentage-based formula makes sense.
- Threshold `θ`: chosen so that the three classes are reasonably balanced **on the training split only** (compute the training split's distribution of `l(t)`, set `θ` to roughly the 33rd/67th percentile split point). **Do not compute θ using validation or test data — that would leak split-specific information into the label definition itself.** Record the exact `θ` used per stock in the results table; it may differ slightly stock to stock, and that's expected and fine, as long as it's derived from that stock's own train split.
- Labels: `down` (label 0) if `l(t) < -θ`, `stable` (label 1) if `-θ ≤ l(t) ≤ θ`, `up` (label 2) if `l(t) > θ`.
- A label is only valid for window-starts where both `t-k+1` and `t+k` exist within the same session (Section 2.3 of Phase 1) — do not compute a label using an average that reaches across a session boundary.

### 2.2 Head architecture

```python
class TrendHead(nn.Module):
    def __init__(self, latent_dim=256, n_classes=3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, n_classes)

    def forward(self, z):  # z: [B, latent_dim]
        return self.fc(z)  # [B, n_classes] logits
```
Deliberately a single linear layer, no hidden layers — the head's job is to test what the frozen latent already encodes, not to add its own representational capacity.

### 2.3 Training protocol

- Loss: `nn.CrossEntropyLoss()`.
- Optimizer: Adam, lr=1e-3 (higher than the encoder's 1e-4 — this is a tiny head with far fewer parameters, and it trains fast; use the full labeled train split of that same stock, not a subsample, since this is same-stock evaluation, not the transfer task in Section 4).
- Epochs: 50 (small head, converges quickly; checkpoint best-val-accuracy, same `save_top_k=1` + `save_last=True` convention as Phase 1).
- Input: the cached latent from Section 1, not a fresh encoder forward pass.

### 2.4 Metrics

Report, per (model, stock): accuracy, macro-F1, and per-class precision/recall. Macro-F1 is the primary metric (accuracy alone is misleading if classes are imbalanced despite the threshold-balancing in 2.1 — report it anyway for interpretability, but rank models by macro-F1).

---

## 3. Task 2 — Imputation

### 3.1 Masking protocol

- Mask a **single contiguous block of 20 consecutive timesteps** (20% of the 100-timestep window — matches the proportion used in the cited literature's imputation task, adopted as a task convention). The block's start position is drawn uniformly at random from `[0, 79]` each time a sample is used (a different mask each epoch, not fixed once — this tests general imputation ability, not memorization of one specific mask pattern).
- **We chose a single contiguous block over scattered random timesteps deliberately**, since the exact masking mechanism in the cited literature's own code was not verified with the same rigor we held everything else in Phase 1 to (see Phase 1 master doc, Section 3.6's TransLOB caveat, for the standard we're holding ourselves to). A single contiguous block is simple, well-defined, and reproducible without ambiguity.
- Masked timesteps are zeroed out in the model input before encoding.

### 3.2 Architecture — reuses the Phase 1 `SharedDecoder` unmodified

Imputation is reconstruction with a partially-corrupted input. The frozen encoder processes the masked window; the same `SharedDecoder` architecture (Section 2.9 of Phase 1) reconstructs the full `[100, 40]` window. **Do not build a separate decoder for this task** — reusing the identical decoder keeps this comparable to the reconstruction numbers from Phase 1 and avoids introducing a second decoder's worth of undocumented capacity, the exact problem Phase 1's Section 2.9 was written to avoid.

Since the encoder is frozen, only the decoder's copy used for this task is trained (a fresh, randomly-initialized `SharedDecoder` instance — not the one trained end-to-end in Phase 1, since that one was optimized jointly with its own encoder on the *unmasked* reconstruction task and testing it here would conflate "does the frozen latent support imputation" with "was this specific decoder instance already tuned for this exact encoder's outputs").

### 3.3 Loss — computed only on masked positions

```python
def imputation_loss(output, target, mask):  # mask: [B, 100] boolean, True = masked position
    diff = (output - target) ** 2
    masked_diff = diff[mask.unsqueeze(-1).expand_as(diff)]
    return masked_diff.mean()
```
**This is important and easy to get wrong:** if loss is computed over the full window (including the 80% that was never masked), the model can trivially achieve a low overall score by just passing through the unmasked positions accurately while doing nothing useful on the actually-imputed ones. Loss and all reported metrics must be restricted to the masked positions only.

### 3.4 Training protocol and metrics

Same optimizer/lr/epoch convention as Section 2.3 (Adam 1e-3, 50 epochs, best-checkpoint by val loss on masked positions). Report MSE and MAE, both computed only on masked positions, per (model, stock).

---

## 4. Task 3 — Cross-Stock Transfer (the headline metric)

This is the one the paper's core positive claim depends on — get this protocol exactly right before running anything.

### 4.1 Protocol

1. **Source stock: sz000001.** The encoder used is the one already trained in Phase 1 on sz000001's train split (no retraining needed — reuse that checkpoint).
2. Freeze that encoder completely.
3. **Target stocks: the other four** (sz000002, sz000858, sz300147, sz002415), one at a time.
4. For each target stock: take **20%** of that target stock's own train split (adopted directly from the cited literature's Table 6 protocol — a deliberate methodological choice, not a number we're citing as a result) to fine-tune a fresh `TrendHead` (Section 2.2). The encoder remains frozen throughout — only the head sees gradients.
5. Evaluate on the target stock's own **test** split (never the 20% used for head training, and never any data used to determine that stock's own trend-labeling threshold `θ` in Section 2.1 — each target stock's `θ` is computed from its own train split independently, same rule as Section 2.1).
6. Task used for the transfer metric: trend prediction (Section 2.1's label definition), matching the classification framing (precision/recall) used in the cited literature's own transfer table — adopted as a protocol choice for direct comparability of *our own* numbers across models, not as a citation of their results.

### 4.2 Metrics

Report precision, recall, and macro-F1 per (model, source→target stock pair). Report the mean across the four target stocks per model as the headline summary number — this is the single number that answers "does this encoder's representation generalize."

### 4.3 Phase 4a (required) vs. Phase 4b (stretch goal)

- **4a, required:** single-source protocol above (sz000001 → each of the other 4). This is what gets run for every model first, and is what goes in the paper's main table.
- **4b, stretch goal, budget-permitting:** the full 5×4 transfer matrix (every stock as source, transferring to every other stock) — 20 pairs instead of 4, repeated per model. Do not start this until 4a is complete and reviewed, and only if remaining Colab budget allows — this is meaningfully more compute (though each individual run is still cheap, since it's just a linear head on cached latents) and more wall-clock/session-management overhead, not a blocker for a first complete draft of results.

---

## 5. Coding rules for this phase

- **Every results report for this phase must include raw evidence, not a narrative summary.** For checkpoint selection: the full printed output of `select_optimal_checkpoint()`, including rejected candidates. For each downstream probe: the actual per-(model, stock) metric values in a table, not a qualitative description of "results look reasonable." For anything claimed to be fixed or verified: the actual before/after output, the same way Section 0a's checkpoint-selection fix was only accepted here after the actual metadata table was reviewed line by line. A report that asserts a result without showing the number or the log it came from will be sent back before it's acted on.

- **New shared module, not a modification of `common.py`.** Create `downstream_common.py` in the same project folder, containing: the latent-caching function (Section 1), `TrendHead` and its training/eval loop (Section 2), the imputation masking + loss + training/eval loop (Section 3), and the transfer protocol orchestration (Section 4). Keep Phase 1's `common.py` untouched — Phase 2 depends on Phase 1's checkpoints but shouldn't risk destabilizing code that's currently mid-run.
- **Notebook structure differs from Phase 1, deliberately.** Phase 1 used one notebook per model because training was expensive and benefited from parallelizing across Google accounts. Phase 2's actual expensive step (running the frozen encoder once per model/stock to produce cached latents) is comparatively cheap, and everything after that is small-head training on tiny cached arrays — plausibly fast enough to run on CPU. Structure Phase 2 as **one notebook per task** (`TrendPrediction.ipynb`, `Imputation.ipynb`, `Transfer.ipynb`), each looping internally over all 6 models (and all 5 stocks, or the relevant subset per Section 4.1). This is a deliberate restructuring from Phase 1's convention, not an inconsistency — the compute profile is genuinely different, so the parallelization strategy should be too.
- Same seed convention as Phase 1 (42, set at the top of every notebook) — head initialization and any masking randomness (Section 3.1) should be reproducible run to run.
- Same checkpoint discipline as Phase 1 (`save_top_k=1`, `save_last=True`) for every head trained, even though these heads are small — resuming after a session interruption should be free by now, not something to reimplement per task.
- Do not start Phase 2 implementation until: (a) all six Phase 1 reconstruction runs are complete and their checkpoints confirmed loadable, and (b) the open item at the top of this document (the checkpoint question) is resolved.

---

## 6. What is intentionally not decided yet

- The full 5×4 transfer matrix (Section 4.3) — deferred until 4a is done and budget is visible.
- TimesNet/iTransformer as a 7th baseline — per the prior discussion, revisited once this phase's results are in and remaining budget is known, not before.
- Whether to add a fine-tuned-encoder ablation alongside the frozen-encoder primary results (Section 1) — worth considering once the frozen results are in hand and show something interesting enough to warrant a "does encoder fine-tuning close the gap" follow-up question.
