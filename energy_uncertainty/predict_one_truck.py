"""Actual vs predicted daily energy for a single truck.

Uses the same features and gradient-boosting setup as predict_energy.py,
trained on all trucks for Jan-Aug, and shows one truck's daily energy across
the year: in-sample fit for Jan-Aug, out-of-sample forecast for Sep-Oct, with
the 5%-95% quantile-boosting interval.

Usage: python predict_one_truck.py [TruckId]   (default 155)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from predict_energy import GB_PARAMS, build_dataset, design

HERE = Path(__file__).resolve().parent
SPLIT = "2023-09-01"


def main(truck_id: int = 155):
    d = build_dataset()
    if truck_id not in set(d["TruckId"]):
        sys.exit(f"Truck {truck_id} not in the data")
    train = d[d["date"] < SPLIT]
    Xtr = design(train)
    ytr = train["energy"]
    models = {
        "pred": GradientBoostingRegressor(**GB_PARAMS),
        "lo": GradientBoostingRegressor(loss="quantile", alpha=0.05, **GB_PARAMS),
        "hi": GradientBoostingRegressor(loss="quantile", alpha=0.95, **GB_PARAMS),
    }
    for m in models.values():
        m.fit(Xtr, ytr)

    t = d[d["TruckId"] == truck_id].sort_values("date").copy()
    Xt = design(t, Xtr.columns)
    for k, m in models.items():
        t[k] = m.predict(Xt)
    info = t.iloc[0]
    test = t[t["date"] >= SPLIT]
    fit = t[t["date"] < SPLIT]

    def stats(s):
        if len(s) < 2:
            return "n/a"
        cov = np.mean((s["energy"] >= s["lo"]) & (s["energy"] <= s["hi"]))
        return (f"R² {r2_score(s['energy'], s['pred']):.2f}, MAE {mean_absolute_error(s['energy'], s['pred']):.1f}\n"
                f"90% coverage {cov:.0%}, n={len(s)} days")

    fig, ax = plt.subplots(figsize=(15, 6), dpi=180)
    ax.axvspan(t["date"].min(), np.datetime64(SPLIT), color="#f2f2f2", lw=0)
    ax.fill_between(t["date"], t["lo"], t["hi"], color="#2a6ebb", alpha=0.18, lw=0, label="90% prediction interval")
    ax.plot(t["date"], t["energy"], color="black", lw=1.2, marker="o", ms=2.5, label="Actual")
    ax.plot(t["date"], t["pred"], color="#2a6ebb", lw=1.8, label="Predicted")
    ax.axvline(np.datetime64(SPLIT), color="#555555", lw=1, ls="--")
    ymax = max(t["energy"].max(), t["hi"].max()) * 1.15
    ax.set_ylim(0, ymax)
    ax.text(np.datetime64("2023-01-04"), ymax * 0.97, f"Training period (in-sample)\n{stats(fit)}", va="top", fontsize=9)
    ax.text(np.datetime64("2023-09-03"), ymax * 0.97, f"Test period\n{stats(test)}", va="top", fontsize=9)

    duty = info["duty_cycle"]
    attrs = (f"{info['truck_category']}, duty {duty}, age {info['age']:.1f} yr, {info['km']:,.0f} km"
             if info["in_fleet"] else f"{info['truck_category']}, not in fleet list")
    ax.set_title(f"Truck {truck_id} — daily energy, actual vs predicted ({attrs})", fontsize=13)
    ax.set_ylabel("Daily energy (dataset units)")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 0.86), ncol=3)
    ax.grid(color="#e5e5e5")
    ax.set_axisbelow(True)
    ax.set_xlim(t["date"].min(), t["date"].max())
    fig.tight_layout()
    out = HERE / f"truck_{truck_id}_energy_prediction.png"
    fig.savefig(out)
    plt.close(fig)
    print(attrs)
    print("train:", stats(fit))
    print("test: ", stats(test))
    print("saved", out.name)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 155)
