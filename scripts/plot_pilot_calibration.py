#!/usr/bin/env python3
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
per_residue = ROOT / "results" / "per_residue.csv"
output = ROOT / "results" / "pilot_plddt_vs_lddt.png"

if not per_residue.exists():
    raise FileNotFoundError(f"Missing per-residue results: {per_residue}")

df = pd.read_csv(per_residue)
df = df.dropna(subset=["pLDDT", "local_lddt"])

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(df["pLDDT"], df["local_lddt"], s=12, alpha=0.6, color="tab:blue")
ax.set_xlabel("pLDDT")
ax.set_ylabel("lDDT")
ax.set_title("Pilot pLDDT vs lDDT")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(output, dpi=200)
print(f"Saved {output}")
