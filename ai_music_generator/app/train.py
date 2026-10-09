import argparse, time
from pathlib import Path
import torch, torch.nn as nn
from .preprocess import load_dataset
from .model import MusicLSTM
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model.pt"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--max-files", type=int, default=150)
    ap.add_argument("--batch", type=int, default=128); ap.add_argument("--hidden", type=int, default=256)
    a = ap.parse_args()
    torch.manual_seed(0)
    X, G, genres = load_dataset(max_files=a.max_files)
    X, G = torch.from_numpy(X), torch.from_numpy(G)
    model = MusicLSTM(len(genres), hidden=a.hidden); opt = torch.optim.Adam(model.parameters(), 2e-3)
    loss_fn = nn.CrossEntropyLoss(); n = len(X); hist = []
    for ep in range(1, a.epochs+1):
        perm, tot, t0 = torch.randperm(n), 0.0, time.time(); model.train()
        for i in range(0, n, a.batch):
            idx = perm[i:i+a.batch]; xb = X[idx]
            logits, _ = model(xb[:, :-1], G[idx])
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), xb[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item() * len(idx)
        hist.append(tot/n); print(f"epoch {ep}/{a.epochs} loss {hist[-1]:.4f} ({time.time()-t0:.0f}s)")
        MODEL_PATH.parent.mkdir(exist_ok=True)
        torch.save({"state": model.state_dict(), "cfg": model.cfg, "genres": genres, "loss": hist, "windows": n}, MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")
if __name__ == "__main__": main()
