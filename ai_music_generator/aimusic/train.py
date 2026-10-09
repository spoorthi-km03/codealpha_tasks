"""Training: fit the conditional LSTM on the tokenised MIDI dataset.

How one training step works
    1. Draw a batch of random windows (default 24 x 192 tokens).  Genres are
       sampled evenly, so a small genre is not drowned by a big one.
    2. Optionally transpose each window by up to +-3 semitones (data augmentation).
    3. Forward pass -> cross-entropy of "predict the next token".
    4. Back-propagation through time -> gradients.
    5. Clip the global gradient norm, then an Adam update with a warm-up +
       cosine learning-rate schedule.

Training is *time-boxed*: you say how many minutes you can spare and the
learning-rate schedule is stretched over exactly that time.
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

from .config import CHECKPOINT_PATH, DATASET_META_PATH, DATASET_PATH, MODEL_PATH
from .lstm import LSTMConfig, LSTMModel
from .tokenizer import (BOS, PAD, VOCAB_SIZE, feasible_transpositions, transpose_tokens)

PRESETS = {
    # name: (minutes, hidden units)
    "quick":    (10, 192),
    "standard": (30, 256),
    "long":     (90, 320),
}


class TrainingError(RuntimeError):
    pass


class TokenDataset:
    """Random-window sampler over the concatenated token array."""

    def __init__(self, npz_path: Path, val_fraction: float = 0.06, seed: int = 0):
        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise TrainingError(
                f"Processed dataset not found: {npz_path}\n"
                "Run  python scripts/02_preprocess.py  first.")
        with np.load(npz_path) as z:
            self.tokens = z["tokens"]
            self.offsets = z["offsets"]
            self.genre_idx = z["genre_idx"].astype(int)
            self.mood_idx = z["mood_idx"].astype(int)
        self.n_files = len(self.genre_idx)
        rng = np.random.default_rng(seed)
        self.train_by_genre: Dict[int, np.ndarray] = {}
        val = []
        for g in sorted(set(self.genre_idx.tolist())):
            files = np.where(self.genre_idx == g)[0]
            files = rng.permutation(files)
            n_val = int(round(len(files) * val_fraction))
            if len(files) >= 6:
                n_val = max(1, n_val)
            else:
                n_val = 0
            val.extend(files[:n_val].tolist())
            self.train_by_genre[g] = files[n_val:]
        self.val_files = np.array(val, dtype=int)
        self.lengths = np.diff(self.offsets)

    def _window(self, f: int, T: int, rng, augment: bool):
        a = self.tokens[self.offsets[f]:self.offsets[f + 1]].astype(np.int64)
        if rng.random() < 0.15:                     # start of piece (with BOS)
            w = np.concatenate([[BOS], a[:T]])
        else:
            s = int(rng.integers(0, max(1, len(a) - T - 1) + 1))
            w = a[s:s + T + 1]
        if len(w) < T + 1:
            w = np.concatenate([w, np.full(T + 1 - len(w), PAD)])
        if augment:
            opts = feasible_transpositions(w, 3)
            shift = int(rng.choice(opts))
            if shift:
                w = transpose_tokens(w, shift)
        return w

    def _make(self, files, T, rng, augment):
        W = np.stack([self._window(f, T, rng, augment) for f in files])
        x, y = W[:, :-1], W[:, 1:]
        mask = (y != PAD).astype(np.float32)
        return (x, y, mask, self.genre_idx[files], self.mood_idx[files])

    def batch(self, B: int, T: int, rng: np.random.Generator):
        genres = list(self.train_by_genre.keys())
        files = []
        for _ in range(B):
            g = genres[int(rng.integers(len(genres)))]
            cand = self.train_by_genre[g]
            p = self.lengths[cand].astype(float)
            files.append(int(rng.choice(cand, p=p / p.sum())))
        return self._make(np.array(files), T, rng, augment=True)

    def val_batches(self, B: int, T: int, n: int = 4):
        if len(self.val_files) == 0:
            return []
        rng = np.random.default_rng(1234)
        out = []
        for _ in range(n):
            files = rng.choice(self.val_files, size=B)
            out.append(self._make(files, T, rng, augment=False))
        return out


class Adam:
    def __init__(self, params, beta1=0.9, beta2=0.999, eps=1e-8):
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t, self.b1, self.b2, self.eps = 0, beta1, beta2, eps

    def step(self, params, grads, lr):
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for k, g in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            params[k] -= (lr * (self.m[k] / c1) / (np.sqrt(self.v[k] / c2) + self.eps)).astype(params[k].dtype)


def _clip(grads, max_norm):
    total = math.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads.values()))
    if total > max_norm:
        s = max_norm / (total + 1e-9)
        for k in grads:
            grads[k] = grads[k] * s
    return total


def _evaluate(model, batches) -> Optional[float]:
    if not batches:
        return None
    losses = [model.loss_and_grads(x, y, m, g, mo, train=False, need_grads=False)[0]
              for x, y, m, g, mo in batches]
    return float(np.mean(losses))


def train(dataset_path: Path = DATASET_PATH, dataset_meta_path: Path = DATASET_META_PATH,
          model_path: Path = MODEL_PATH, checkpoint_path: Path = CHECKPOINT_PATH,
          minutes: float = 30, hidden: int = 256, layers: int = 2, batch_size: int = 24,
          seq_len: int = 192, lr: float = 2e-3, max_steps: Optional[int] = None,
          resume: bool = False, seed: int = 0, log: Callable[[str], None] = print) -> dict:
    dataset_meta_path = Path(dataset_meta_path)
    if not dataset_meta_path.exists():
        raise TrainingError(f"{dataset_meta_path} not found - run scripts/02_preprocess.py first.")
    dmeta = json.loads(dataset_meta_path.read_text(encoding="utf-8"))
    ds = TokenDataset(dataset_path, seed=seed)
    if ds.tokens.size < seq_len * 4:
        raise TrainingError("The dataset is too small to train on - add more MIDI files.")

    cfg = LSTMConfig(vocab_size=VOCAB_SIZE, n_genres=len(dmeta["genres"]),
                     n_moods=len(dmeta["moods"]), hidden=hidden, layers=layers)
    model = LSTMModel(cfg, seed=seed)
    opt = Adam(model.params)
    step, elapsed_before = 0, 0.0
    checkpoint_path = Path(checkpoint_path)
    if resume and checkpoint_path.exists():
        with np.load(checkpoint_path, allow_pickle=False) as z:
            saved_cfg = json.loads(str(z["config_json"]))
            if saved_cfg != json.loads(json.dumps(cfg.__dict__)):
                raise TrainingError("Checkpoint architecture differs from the requested one; "
                                    "train without --resume or use the same settings.")
            for k in model.params:
                model.params[k][...] = z[f"p_{k}"]
                opt.m[k][...] = z[f"m_{k}"]
                opt.v[k][...] = z[f"v_{k}"]
            step, opt.t, elapsed_before = int(z["step"]), int(z["adam_t"]), float(z["elapsed"])
        log(f"Resumed from checkpoint at step {step} ({elapsed_before / 60:.1f} min already trained)")
    elif resume:
        log("No checkpoint found - starting from scratch.")

    log(f"Model: {model.n_parameters():,} parameters | hidden={hidden} layers={layers}")
    log(f"Data : {ds.n_files} files, {ds.tokens.size:,} tokens, genres={dmeta['genres']}, "
        f"moods={dmeta['moods']}, validation files={len(ds.val_files)}")
    budget = f"{minutes} min" if max_steps is None else f"{max_steps} steps"
    log(f"Training budget: {budget}.  Press Ctrl+C at any time to stop and keep the model so far.\n")

    rng = np.random.default_rng(seed + step)
    val_batches = ds.val_batches(batch_size, seq_len)
    t_start = time.time()
    last_ckpt = t_start
    recent, best_val, best_params = [], None, None
    last_val_step = -1
    total_tokens = 0
    interrupted = False

    def elapsed():
        return elapsed_before + (time.time() - t_start)

    def progress():
        if max_steps is not None:
            return min(1.0, step / max_steps)
        return min(1.0, elapsed() / (minutes * 60))

    def save_checkpoint():
        data = {"config_json": np.array(json.dumps(cfg.__dict__)), "step": np.array(step),
                "adam_t": np.array(opt.t), "elapsed": np.array(elapsed())}
        for k in model.params:
            data[f"p_{k}"], data[f"m_{k}"], data[f"v_{k}"] = model.params[k], opt.m[k], opt.v[k]
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(checkpoint_path, **data)

    try:
        while progress() < 1.0:
            f = progress()
            warm = min(1.0, (step + 1) / 30)
            cur_lr = lr * warm * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * f)))
            x, y, mask, g, mo = ds.batch(batch_size, seq_len, rng)
            loss, grads = model.loss_and_grads(x, y, mask, g, mo, train=True, rng=rng)
            if not math.isfinite(loss):
                raise TrainingError("Loss became NaN/inf - try a lower learning rate (--lr 0.001).")
            _clip(grads, 5.0)
            opt.step(model.params, grads, cur_lr)
            step += 1
            total_tokens += int(mask.sum())
            recent.append(loss)
            if step % 50 == 0:
                avg = float(np.mean(recent[-50:]))
                done = progress()
                eta = (elapsed() - elapsed_before) * (1 - done) / max(done - 0, 1e-9) if max_steps else \
                    max(0.0, minutes * 60 - elapsed())
                log(f"step {step:6d} | loss {avg:.3f} | perplexity {math.exp(avg):7.1f} | "
                    f"lr {cur_lr:.5f} | {done * 100:5.1f}% | ~{eta / 60:4.1f} min left")
            if step % 250 == 0 and val_batches:
                v = _evaluate(model, val_batches)
                last_val_step = step
                log(f"          validation loss {v:.3f} (perplexity {math.exp(v):.1f})")
                if best_val is None or v < best_val:
                    best_val = v
                    best_params = {k: a.copy() for k, a in model.params.items()}
            if time.time() - last_ckpt > 120:
                save_checkpoint()
                last_ckpt = time.time()
    except KeyboardInterrupt:
        interrupted = True
        log("\nInterrupted - saving what has been learned so far...")

    if step == 0:
        raise TrainingError("No training step was completed.")
    final_val = _evaluate(model, val_batches)
    if best_params is not None and final_val is not None and best_val is not None and best_val < final_val:
        log(f"Using the parameters with the best validation loss ({best_val:.3f} < final {final_val:.3f}).")
        model.params = best_params
        final_val = best_val
    final_train = float(np.mean(recent[-100:]))
    save_checkpoint()

    meta = {
        "trained": True,
        "genres": dmeta["genres"],
        "moods": dmeta["moods"],
        "combo_counts": dmeta["combo_counts"],
        "files_per_genre": dmeta["files_per_genre"],
        "steps_per_beat": dmeta["steps_per_beat"],
        "mood_method": dmeta["mood_method"],
        "training": {
            "steps": step, "minutes": round(elapsed() / 60, 2), "batch_size": batch_size,
            "seq_len": seq_len, "learning_rate": lr, "interrupted": interrupted,
            "train_loss": round(final_train, 4), "val_loss": None if final_val is None else round(final_val, 4),
            "train_perplexity": round(math.exp(final_train), 2),
            "val_perplexity": None if final_val is None else round(math.exp(final_val), 2),
            "random_guess_loss": round(math.log(VOCAB_SIZE), 4),
            "dataset_files": dmeta["n_files"], "dataset_tokens": dmeta["n_tokens"],
            "trained_at": datetime.now().isoformat(timespec="seconds"),
        },
    }
    model.save(model_path, meta)
    log(f"\nSaved model -> {model_path}")
    log(f"Final train loss {final_train:.3f} (random guessing would be {math.log(VOCAB_SIZE):.3f})"
        + ("" if final_val is None else f", validation loss {final_val:.3f}"))
    return meta
