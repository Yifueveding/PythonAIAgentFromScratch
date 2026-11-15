"""Predict daily energy per truck from truck attributes, distance and temperature.

Target: energy of one truck on one active calendar day (sum of its half-hour slots).

Features
  truck-level : truck_category, duty cycle, age (yr), odometer (km)
  day-level   : fleet total distance (Total_Distance.csv), daily mean temperature
                (temperature.csv), month

Trucks missing from West_Ang_Fleet.csv (all Ghosts, a few Mobile_Eq) get duty
cycle "UNKNOWN", median age/km and an ``in_fleet`` = 0 flag.

Caveats: temperature.csv is 2019 weather while the energy data appears to be
2023, so temperature is joined on month and day and acts mostly as a seasonal
signal. Distance is a fleet-wide total, not the truck's own distance.

Validation
  time split  : train Jan-Aug, test Sep-Oct (forecasting forward)
  truck split : 5-fold GroupKFold on TruckId (trucks never seen in training)
Models: category-mean baseline, per-truck-mean baseline (time split only),
linear regression, gradient boosting; plus 5%/95% quantile boosting for a 90%
prediction interval.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

from energy_data import load_energy

HERE = Path(__file__).resolve().parent
OTHER = HERE.parent / "Other_data"
SEED = 0
GB_PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=SEED)

NUMERIC = ["age", "km_k", "in_fleet", "distance_k", "temp", "month"]
CATEGORICAL = ["truck_category", "duty_cycle"]
FEATURE_LABELS = {
    "truck_category": "Truck category", "duty_cycle": "Duty cycle", "age": "Age",
    "km_k": "Odometer (km)", "in_fleet": "In fleet list", "distance_k": "Fleet distance",
    "temp": "Temperature", "month": "Month",
}


def build_dataset() -> pd.DataFrame:
    e = load_energy()
    d = e.groupby(["TruckId", "truck_category", "date"], as_index=False)["energy_consumption"].sum()
    d = d.rename(columns={"energy_consumption": "energy"})
    d["md"] = d["date"].dt.strftime("%m-%d")
    d["month"] = d["date"].dt.month

    fleet = pd.read_csv(OTHER / "West_Ang_Fleet.csv", usecols=["VehicleId", "Age", "Km's", "Duty Cycle"])
    fleet = fleet.dropna(subset=["VehicleId"]).rename(columns={
        "VehicleId": "TruckId", "Age": "age", "Km's": "km", "Duty Cycle": "duty_cycle"})
    fleet["TruckId"] = fleet["TruckId"].astype(int)
    d = d.merge(fleet, on="TruckId", how="left")
    d["in_fleet"] = d["age"].notna().astype(int)
    d["duty_cycle"] = d["duty_cycle"].fillna("UNKNOWN")
    for c in ["age", "km"]:
        d[c] = d[c].fillna(fleet.loc[fleet["TruckId"].isin(d["TruckId"]), c].median())
    d["km_k"] = d["km"] / 1000

    dist = pd.read_csv(OTHER / "Total_Distance.csv", parse_dates=["Date"])
    dist = pd.Series(dist["Total Distance"].to_numpy() / 1000, index=dist["Date"].dt.strftime("%m-%d"), name="distance_k")
    t = pd.read_csv(OTHER / "temperature.csv", comment="#", parse_dates=["local_time"])
    temp = t.groupby(t["local_time"].dt.strftime("%m-%d"))["t2m"].mean().rename("temp")
    d = d.join(dist, on="md").join(temp, on="md")
    return d.dropna(subset=["distance_k", "temp"]).reset_index(drop=True)


def design(d: pd.DataFrame, columns=None) -> pd.DataFrame:
    X = pd.get_dummies(d[NUMERIC + CATEGORICAL], columns=CATEGORICAL, dtype=float)
    return X if columns is None else X.reindex(columns=columns, fill_value=0.0)


def fit_predict(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    Xtr = design(train)
    Xte = design(test, Xtr.columns)
    ytr = train["energy"]
    out = {"category mean": test["truck_category"].map(train.groupby("truck_category")["energy"].mean()).to_numpy()}
    seen = test["TruckId"].isin(train["TruckId"])
    if seen.all():
        out["per-truck mean"] = test["TruckId"].map(train.groupby("TruckId")["energy"].mean()).to_numpy()
    out["linear regression"] = LinearRegression().fit(Xtr, ytr).predict(Xte)
    gb = GradientBoostingRegressor(**GB_PARAMS).fit(Xtr, ytr)
    out["gradient boosting"] = gb.predict(Xte)
    out["_lo"] = GradientBoostingRegressor(loss="quantile", alpha=0.05, **GB_PARAMS).fit(Xtr, ytr).predict(Xte)
    out["_hi"] = GradientBoostingRegressor(loss="quantile", alpha=0.95, **GB_PARAMS).fit(Xtr, ytr).predict(Xte)
    out["_gb"] = gb
    out["_X"] = Xte
    return out


def scores(y, preds: dict) -> pd.DataFrame:
    rows = [(k, r2_score(y, p), mean_absolute_error(y, p)) for k, p in preds.items() if not k.startswith("_")]
    return pd.DataFrame(rows, columns=["model", "R2", "MAE"]).set_index("model")


def grouped_importance(model, X: pd.DataFrame, y) -> pd.Series:
    """Permutation importance with one-hot columns permuted together per original feature."""
    rng = np.random.default_rng(SEED)
    base = r2_score(y, model.predict(X))
    groups = {f: [c for c in X.columns if c == f or c.startswith(f + "_")] for f in NUMERIC + CATEGORICAL}
    imp = {}
    for f, cols in groups.items():
        drops = []
        for _ in range(5):
            Xp = X.copy()
            Xp[cols] = Xp[cols].to_numpy()[rng.permutation(len(Xp))]
            drops.append(base - r2_score(y, model.predict(Xp)))
        imp[FEATURE_LABELS[f]] = np.mean(drops)
    return pd.Series(imp).sort_values()


def main():
    d = build_dataset()
    print(f"{len(d)} truck-days, {d['TruckId'].nunique()} trucks, "
          f"{d['date'].min():%Y-%m-%d} to {d['date'].max():%Y-%m-%d}")

    # --- time split -------------------------------------------------------
    is_test = d["date"] >= "2023-09-01"
    train, test = d[~is_test], d[is_test]
    test = test[test["TruckId"].isin(train["TruckId"])]  # per-truck baseline needs history
    tp = fit_predict(train, test)
    time_scores = scores(test["energy"], tp)
    coverage = np.mean((test["energy"] >= tp["_lo"]) & (test["energy"] <= tp["_hi"]))
    print("\nTime split (train Jan-Aug, test Sep-Oct):\n", time_scores.round(3))
    print(f"90% interval coverage: {coverage:.1%}, mean width {np.mean(tp['_hi'] - tp['_lo']):.1f}")

    # --- truck split (unseen trucks) --------------------------------------
    oof = {}
    for tr_idx, te_idx in GroupKFold(n_splits=5).split(d, groups=d["TruckId"]):
        p = fit_predict(d.iloc[tr_idx], d.iloc[te_idx])
        for k in ["category mean", "linear regression", "gradient boosting", "_lo", "_hi"]:
            oof.setdefault(k, np.empty(len(d)))[te_idx] = p[k]
    truck_scores = scores(d["energy"], oof)
    truck_cov = np.mean((d["energy"] >= oof["_lo"]) & (d["energy"] <= oof["_hi"]))
    print("\nTruck split (5-fold GroupKFold, unseen trucks):\n", truck_scores.round(3))
    print(f"90% interval coverage: {truck_cov:.1%}")

    imp = grouped_importance(tp["_gb"], tp["_X"], test["energy"])
    print("\nPermutation importance (drop in R2, time-split test set):\n", imp[::-1].round(3))

    # --- figure -----------------------------------------------------------
    fig = plt.figure(figsize=(16, 11), dpi=170)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
    ax_sc, ax_imp, ax_tab = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    ax_ts = fig.add_subplot(gs[1, :])

    y, yhat = test["energy"].to_numpy(), tp["gradient boosting"]
    ax_sc.scatter(y, yhat, s=4, alpha=0.25, color="#2a6ebb", lw=0)
    lim = np.percentile(np.r_[y, yhat], 99.5)
    ax_sc.plot([0, lim], [0, lim], color="black", lw=1, ls="--")
    ax_sc.set_xlim(0, lim)
    ax_sc.set_ylim(0, lim)
    ax_sc.set_xlabel("Actual daily energy per truck")
    ax_sc.set_ylabel("Predicted (gradient boosting)")
    ax_sc.set_title(f"(a) Truck-days, Sep–Oct test set\nR² = {r2_score(y, yhat):.2f}, MAE = {mean_absolute_error(y, yhat):.1f}", fontsize=11)
    ax_sc.grid(color="#e5e5e5")
    ax_sc.set_axisbelow(True)

    ax_imp.barh(imp.index, imp.to_numpy(), color="#7e5fa6")
    ax_imp.set_xlabel("Drop in R² when shuffled")
    ax_imp.set_title("(b) Permutation feature importance", fontsize=11)
    ax_imp.grid(axis="x", color="#e5e5e5")
    ax_imp.set_axisbelow(True)

    ax_tab.axis("off")
    tab = pd.concat({"Time split": time_scores, "Unseen trucks": truck_scores}, axis=1)
    cells = [[m] + [("—" if pd.isna(v) else f"{v:.2f}" if "R2" in c else f"{v:.1f}") for c, v in zip(tab.columns, row)]
             for m, row in zip(tab.index, tab.to_numpy())]
    cells.append(["90% interval coverage", f"{coverage:.0%}", "", f"{truck_cov:.0%}", ""])
    t = ax_tab.table(cellText=cells, colLabels=["Model", "R²\ntime", "MAE\ntime", "R²\ntrucks", "MAE\ntrucks"],
                     loc="center", cellLoc="center", colWidths=[0.42, 0.15, 0.15, 0.15, 0.15])
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1.1, 1.6)
    ax_tab.set_title("(c) Accuracy by model and validation", fontsize=11)

    # Fleet-level 90% interval: per-truck intervals do not add up, so use the
    # 5th/95th percentiles of fleet daily-total residuals on the training months
    # (out-of-fold over trucks, so they are not optimistic in-sample residuals).
    train_fleet = d[~is_test].assign(pred=oof["gradient boosting"][~is_test.to_numpy()]).groupby("date")[["energy", "pred"]].sum()
    q_lo, q_hi = np.percentile(train_fleet["energy"] - train_fleet["pred"], [5, 95])
    fleet = test.assign(pred=yhat).groupby("date")[["energy", "pred"]].sum()
    fleet["lo"], fleet["hi"] = fleet["pred"] + q_lo, fleet["pred"] + q_hi
    fleet_r2 = r2_score(fleet["energy"], fleet["pred"])
    fleet_cov = np.mean((fleet["energy"] >= fleet["lo"]) & (fleet["energy"] <= fleet["hi"]))
    fleet_mape = np.mean(np.abs(fleet["energy"] - fleet["pred"]) / fleet["energy"])
    ax_ts.fill_between(fleet.index, fleet["lo"], fleet["hi"], color="#2a6ebb", alpha=0.15, lw=0,
                       label=f"90% interval (coverage {fleet_cov:.0%})")
    ax_ts.plot(fleet.index, fleet["energy"], color="black", lw=1.5, label="Actual")
    ax_ts.plot(fleet.index, fleet["pred"], color="#2a6ebb", lw=1.8, label="Predicted")
    ax_ts.set_ylabel("Fleet total energy per day")
    ax_ts.set_title(f"(d) Fleet daily total, Sep–Oct test period (R² = {fleet_r2:.2f}, "
                    f"mean abs. error {fleet_mape:.1%})", fontsize=11)
    ax_ts.legend(frameon=False, loc="upper left", ncol=3)
    ax_ts.grid(color="#e5e5e5")
    ax_ts.set_axisbelow(True)
    ax_ts.set_ylim(fleet[["energy", "lo"]].min().min() * 0.9, fleet[["energy", "hi"]].max().max() * 1.08)

    fig.tight_layout()
    fig.savefig(HERE / "energy_prediction.png")
    plt.close(fig)
    print(f"\nFleet daily total (Sep-Oct): R2 {fleet_r2:.3f}, MAPE {fleet_mape:.2%}, "
          f"90% interval coverage {fleet_cov:.0%} (residual band {q_lo:.0f} to +{q_hi:.0f})")


if __name__ == "__main__":
    main()
