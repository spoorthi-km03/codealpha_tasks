"""A conditional, stacked LSTM language model written from scratch in NumPy.

Why NumPy instead of TensorFlow / PyTorch?
    * the only heavy dependency is NumPy, which installs instantly on
      Windows 11 + Python 3.12;
    * every equation is visible, which makes the model easy to explain;
    * it runs on any CPU - no GPU, no CUDA, no large downloads.

Architecture (per time step t)

    x_t   = [ tokenEmb(token_t) | genreEmb(genre) | moodEmb(mood) ]
    layer 1: LSTM(x_t)            -> h1_t  -> dropout
    layer 2: LSTM(h1_t)           -> h2_t  -> dropout
    logits_t = h2_t @ Wo + bo     -> softmax over the next token

The genre and mood embeddings are fed at *every* step, so the conditioning
cannot be "forgotten" during a long piece.

LSTM cell (gates stacked in the order  i, f, g, o):
    z  = x_t Wx + h_{t-1} Wh + b
    i  = sigmoid(z_i)   f = sigmoid(z_f)   g = tanh(z_g)   o = sigmoid(z_o)
    c_t = f * c_{t-1} + i * g
    h_t = o * tanh(c_t)

Training uses back-propagation through time (BPTT) implemented by hand; the
test-suite verifies the gradients against numerical finite differences.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class LSTMConfig:
    vocab_size: int
    n_genres: int
    n_moods: int
    d_token: int = 48
    d_cond: int = 8
    hidden: int = 192
    layers: int = 2
    dropout: float = 0.1
    dtype: str = "float32"

    @property
    def d_input(self) -> int:
        return self.d_token + 2 * self.d_cond


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class LSTMModel:
    def __init__(self, config: LSTMConfig, params: Optional[Dict[str, np.ndarray]] = None,
                 seed: int = 0):
        self.config = config
        self.params: Dict[str, np.ndarray] = params if params is not None else self._init(seed)

    # ------------------------------------------------------------------ init
    def _init(self, seed: int) -> Dict[str, np.ndarray]:
        c = self.config
        rng = np.random.default_rng(seed)
        dt = np.dtype(c.dtype)
        p: Dict[str, np.ndarray] = {}
        p["E_tok"] = rng.normal(0, 0.1, (c.vocab_size, c.d_token))
        p["E_genre"] = rng.normal(0, 0.1, (c.n_genres, c.d_cond))
        p["E_mood"] = rng.normal(0, 0.1, (c.n_moods, c.d_cond))
        in_dim = c.d_input
        for l in range(c.layers):
            H = c.hidden
            p[f"Wx{l}"] = rng.normal(0, 1.0 / np.sqrt(in_dim), (in_dim, 4 * H))
            p[f"Wh{l}"] = rng.normal(0, 1.0 / np.sqrt(H), (H, 4 * H))
            b = np.zeros(4 * H)
            b[H:2 * H] = 1.0            # forget-gate bias = 1 (helps remembering)
            p[f"b{l}"] = b
            in_dim = H
        p["Wo"] = rng.normal(0, 1.0 / np.sqrt(c.hidden), (c.hidden, c.vocab_size))
        p["bo"] = np.zeros(c.vocab_size)
        return {k: v.astype(dt) for k, v in p.items()}

    def n_parameters(self) -> int:
        return int(sum(v.size for v in self.params.values()))

    # --------------------------------------------------------------- helpers
    def _inputs(self, tokens: np.ndarray, genre: np.ndarray, mood: np.ndarray) -> np.ndarray:
        """tokens (B,T) -> input tensor (T,B,d_input)."""
        P = self.params
        B, T = tokens.shape
        tok = P["E_tok"][tokens]                                  # (B,T,d_tok)
        g = np.repeat(P["E_genre"][genre][:, None, :], T, axis=1)
        m = np.repeat(P["E_mood"][mood][:, None, :], T, axis=1)
        x = np.concatenate([tok, g, m], axis=2)
        return np.ascontiguousarray(x.transpose(1, 0, 2))

    @staticmethod
    def _lstm_forward(X, Wx, Wh, b, h0, c0):
        T, B, _ = X.shape
        H = Wh.shape[0]
        Zx = (X.reshape(T * B, -1) @ Wx + b).reshape(T, B, 4 * H)
        dt = Zx.dtype
        Hs = np.empty((T, B, H), dt)
        Cs = np.empty((T, B, H), dt)
        I = np.empty((T, B, H), dt)
        F = np.empty((T, B, H), dt)
        G = np.empty((T, B, H), dt)
        O = np.empty((T, B, H), dt)
        TC = np.empty((T, B, H), dt)
        h, c = h0, c0
        for t in range(T):
            z = Zx[t] + h @ Wh
            i = _sigmoid(z[:, :H])
            f = _sigmoid(z[:, H:2 * H])
            g = np.tanh(z[:, 2 * H:3 * H])
            o = _sigmoid(z[:, 3 * H:])
            c = f * c + i * g
            tc = np.tanh(c)
            h = o * tc
            Hs[t], Cs[t], I[t], F[t], G[t], O[t], TC[t] = h, c, i, f, g, o, tc
        return Hs, (Cs, I, F, G, O, TC)

    @staticmethod
    def _lstm_backward(dHs, cache, X, Wx, Wh, h0, c0, Hs):
        Cs, I, F, G, O, TC = cache
        T, B, H = dHs.shape
        dt = dHs.dtype
        dZ = np.empty((T, B, 4 * H), dt)
        dh_next = np.zeros((B, H), dt)
        dc_next = np.zeros((B, H), dt)
        for t in range(T - 1, -1, -1):
            dh = dHs[t] + dh_next
            c_prev = Cs[t - 1] if t > 0 else c0
            do = dh * TC[t]
            dc = dh * O[t] * (1.0 - TC[t] ** 2) + dc_next
            di = dc * G[t]
            dg = dc * I[t]
            df = dc * c_prev
            dc_next = dc * F[t]
            dZ[t, :, :H] = di * I[t] * (1.0 - I[t])
            dZ[t, :, H:2 * H] = df * F[t] * (1.0 - F[t])
            dZ[t, :, 2 * H:3 * H] = dg * (1.0 - G[t] ** 2)
            dZ[t, :, 3 * H:] = do * O[t] * (1.0 - O[t])
            dh_next = dZ[t] @ Wh.T
        Hprev = np.concatenate([h0[None], Hs[:-1]], axis=0)
        dZ2 = dZ.reshape(T * B, 4 * H)
        dWx = X.reshape(T * B, -1).T @ dZ2
        dWh = Hprev.reshape(T * B, H).T @ dZ2
        db = dZ2.sum(axis=0)
        dX = (dZ2 @ Wx.T).reshape(T, B, -1)
        return dX, dWx, dWh, db

    # --------------------------------------------------------------- training
    def loss_and_grads(self, tokens: np.ndarray, targets: np.ndarray, mask: np.ndarray,
                       genre: np.ndarray, mood: np.ndarray, train: bool = True,
                       rng: Optional[np.random.Generator] = None,
                       need_grads: bool = True):
        """Cross-entropy of next-token prediction (+ gradients via BPTT).

        tokens, targets, mask: (B,T); genre, mood: (B,).  ``mask`` is 1.0 where the
        position counts towards the loss.
        """
        c, P = self.config, self.params
        dt = np.dtype(c.dtype)
        B, T = tokens.shape
        H = c.hidden
        rng = rng or np.random.default_rng()

        X0 = self._inputs(tokens, genre, mood)
        layer_in = [X0]
        caches, Hs_list, drop_masks = [], [], []
        h0 = np.zeros((B, H), dt)
        c0 = np.zeros((B, H), dt)
        x = X0
        for l in range(c.layers):
            Hs, cache = self._lstm_forward(x, P[f"Wx{l}"], P[f"Wh{l}"], P[f"b{l}"], h0, c0)
            caches.append(cache)
            Hs_list.append(Hs)
            if train and c.dropout > 0:
                dm = ((rng.random(Hs.shape) >= c.dropout) / (1.0 - c.dropout)).astype(dt)
            else:
                dm = None
            drop_masks.append(dm)
            x = Hs * dm if dm is not None else Hs
            layer_in.append(x)

        top = x                                                   # (T,B,H)
        logits = top.reshape(T * B, H) @ P["Wo"] + P["bo"]
        logits -= logits.max(axis=1, keepdims=True)
        expl = np.exp(logits)
        probs = expl / expl.sum(axis=1, keepdims=True)            # (T*B,V)
        tgt = targets.T.reshape(-1)
        msk = mask.T.reshape(-1).astype(dt)
        denom = max(float(msk.sum()), 1.0)
        logp = np.log(probs[np.arange(T * B), tgt] + 1e-12)
        loss = float(-(logp * msk).sum() / denom)
        if not need_grads:
            return loss, None

        # ---- backward ----
        dlogits = probs
        dlogits[np.arange(T * B), tgt] -= 1.0
        dlogits *= (msk / denom)[:, None]
        grads: Dict[str, np.ndarray] = {}
        grads["Wo"] = top.reshape(T * B, H).T @ dlogits
        grads["bo"] = dlogits.sum(axis=0)
        dtop = (dlogits @ P["Wo"].T).reshape(T, B, H)
        d_out = dtop
        for l in range(c.layers - 1, -1, -1):
            if drop_masks[l] is not None:
                d_out = d_out * drop_masks[l]
            dX, dWx, dWh, db = self._lstm_backward(
                d_out, caches[l], layer_in[l], P[f"Wx{l}"], P[f"Wh{l}"], h0, c0, Hs_list[l])
            grads[f"Wx{l}"], grads[f"Wh{l}"], grads[f"b{l}"] = dWx, dWh, db
            d_out = dX
        # d_out is now the gradient w.r.t. the concatenated embedding input (T,B,d_in)
        dtok = d_out[:, :, :c.d_token]
        dg = d_out[:, :, c.d_token:c.d_token + c.d_cond].sum(axis=0)     # (B,d_cond)
        dm_ = d_out[:, :, c.d_token + c.d_cond:].sum(axis=0)
        onehot = np.zeros((B * T, c.vocab_size), dt)
        onehot[np.arange(B * T), tokens.reshape(-1)] = 1.0
        grads["E_tok"] = onehot.T @ dtok.transpose(1, 0, 2).reshape(B * T, -1)
        gE = np.zeros_like(P["E_genre"])
        np.add.at(gE, genre, dg)
        mE = np.zeros_like(P["E_mood"])
        np.add.at(mE, mood, dm_)
        grads["E_genre"], grads["E_mood"] = gE, mE
        return loss, grads

    # -------------------------------------------------------------- inference
    def initial_state(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        H, dt = self.config.hidden, np.dtype(self.config.dtype)
        return [(np.zeros((1, H), dt), np.zeros((1, H), dt)) for _ in range(self.config.layers)]

    def step(self, token: int, genre: int, mood: int, state):
        """Feed ONE token; return (logits for the next token, new state)."""
        c, P = self.config, self.params
        H = c.hidden
        x = np.concatenate([P["E_tok"][token], P["E_genre"][genre], P["E_mood"][mood]])[None, :]
        new_state = []
        for l in range(c.layers):
            h, cell = state[l]
            z = x @ P[f"Wx{l}"] + h @ P[f"Wh{l}"] + P[f"b{l}"]
            i = _sigmoid(z[:, :H])
            f = _sigmoid(z[:, H:2 * H])
            g = np.tanh(z[:, 2 * H:3 * H])
            o = _sigmoid(z[:, 3 * H:])
            cell = f * cell + i * g
            h = o * np.tanh(cell)
            new_state.append((h, cell))
            x = h
        logits = x @ P["Wo"] + P["bo"]
        return logits[0].astype(np.float64), new_state

    # ----------------------------------------------------------- persistence
    def save(self, path: Path, meta: Optional[dict] = None) -> None:
        """Weights go to ``path`` (.npz); metadata (JSON) goes next to it."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.params)
        meta_path = path.with_suffix(".json")
        payload = dict(meta or {})
        payload["architecture"] = asdict(self.config)
        payload["n_parameters"] = self.n_parameters()
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Tuple["LSTMModel", dict]:
        path = Path(path)
        meta_path = path.with_suffix(".json")
        if not path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"model files not found: {path} / {meta_path.name}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cfg = LSTMConfig(**meta["architecture"])
        with np.load(path, allow_pickle=False) as z:
            params = {k: z[k].astype(np.dtype(cfg.dtype)) for k in z.files}
        model = cls(cfg, params)
        expected = cls(cfg, seed=0).params
        for k, v in expected.items():
            if k not in params or params[k].shape != v.shape:
                raise ValueError(f"model file is corrupt or incompatible (parameter '{k}')")
        return model, meta
