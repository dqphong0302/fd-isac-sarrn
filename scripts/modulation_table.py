#!/usr/bin/env python3
"""modulation_table.py — BER of QPSK / 16-QAM runs relative to perfect CSI (S2 Table).

  python scripts/modulation_table.py --results results

Reads results/mod_{qpsk,16qam}/frames.csv (from eval_channel.py --mod ...), writes table.csv next
to them (pooled BER, 95 % Clopper-Pearson interval) and prints the relative BER increase vs genie.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    a = ap.parse_args()
    for mod, k in (("qpsk", 2), ("16qam", 4)):
        d = pd.read_csv(os.path.join(a.results, f"mod_{mod}", "frames.csv"))
        d = d[d.fb == 2]
        rows = []
        for (Np, m, g), x in d.groupby(["np", "method", "gamma_db"]):
            nb = (512 - Np) * 1024 * k * len(x)
            e = int(round((x.ber * (512 - Np) * 1024 * k).sum()))
            rows.append(dict(np=Np, method=m, gamma_db=g, frames=len(x),
                             nmse_db=10 * np.log10(x.nmse.mean()) if m != "genie" else np.nan,
                             bits=nb, bit_errors=e, ber=e / nb,
                             ber_lo=beta.ppf(.025, e, nb - e + 1) if e else 0.0, ber_hi=beta.ppf(.975, e + 1, nb - e)))
        T = pd.DataFrame(rows)
        T.to_csv(os.path.join(a.results, f"mod_{mod}", "table.csv"), index=False)
        G = T[(T.np == 1) & (T.method == "genie")].set_index("gamma_db")
        print("==", mod)
        for g in sorted(T.gamma_db.unique()):
            gb = G.loc[g, "ber"]
            s = f"g={g:4.0f} genie={gb:.3e} | "
            for Np, m in ((1, "ls"), (2, "ls"), (1, "ddls"), (1, "sarrn_da")):
                r = T[(T.np == Np) & (T.method == m) & (T.gamma_db == g)].iloc[0]
                rel = (r.ber / gb - 1) * 100 if gb > 0 else float("nan")
                s += f"{m}{Np}: {r.nmse_db:6.1f} {r.ber:.2e}({rel:+.1f}%) "
            print(s)


if __name__ == "__main__":
    main()
