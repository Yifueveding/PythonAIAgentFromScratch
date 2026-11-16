"""Build fixed-extent route rasters per truck-day from raw GPS positions.

Unlike the Vehicle/*.png route images (each scaled to fit its own route), all
days share the same map extent, so a raster keeps where the truck drove and
how far it ranged. Cells are binary (visited while moving or not), so the
number of laps over the same road is NOT encoded: the raster describes the
route, not the distance driven.

Grids
  fine   : 64 x 64 over the site, centre (118.76, -23.16) +/- 0.15 deg (~0.5 km cells)
  coarse : 32 x 32 over the region, lon 117.3-119.3, lat -23.3 to -22.35

Output: route_rasters.npz with arrays truck, date (YYYY-MM-DD), fine_idx/coarse_idx
(flattened visited-cell indices, ragged via offsets).
"""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
POSITIONS = HERE.parent / "Positions"
OUT = HERE / "route_rasters.npz"

FINE_N, COARSE_N = 64, 32
FINE_BOX = (118.76 - 0.15, 118.76 + 0.15, -23.16 - 0.15, -23.16 + 0.15)  # lon0, lon1, lat0, lat1
COARSE_BOX = (117.3, 119.3, -23.3, -22.35)


def cells(lon, lat, box, n):
    lon0, lon1, lat0, lat1 = box
    ok = (lon >= lon0) & (lon < lon1) & (lat >= lat0) & (lat < lat1)
    ix = ((lon[ok] - lon0) / (lon1 - lon0) * n).astype(int)
    iy = ((lat[ok] - lat0) / (lat1 - lat0) * n).astype(int)
    return ok, iy * n + ix


def main():
    trucks, dates, fine, coarse = [], [], [], []
    files = sorted(POSITIONS.glob("2023-*.csv"))
    for i, f in enumerate(files, 1):
        df = pd.read_csv(f, usecols=["VehicleId", "Longitude", "Latitude", "Speed"])
        df = df[df["Speed"] > 0]
        for vid, g in df.groupby("VehicleId"):
            lon, lat = g["Longitude"].to_numpy(), g["Latitude"].to_numpy()
            _, fi = cells(lon, lat, FINE_BOX, FINE_N)
            _, ci = cells(lon, lat, COARSE_BOX, COARSE_N)
            trucks.append(int(vid))
            dates.append(f.stem)
            fine.append(np.unique(fi).astype(np.int16))
            coarse.append(np.unique(ci).astype(np.int16))
        if i % 30 == 0:
            print(f"  {i}/{len(files)} days", flush=True)

    def pack(parts):
        offsets = np.r_[0, np.cumsum([len(p) for p in parts])]
        return np.concatenate(parts) if parts else np.array([], np.int16), offsets

    fv, fo = pack(fine)
    cv, co = pack(coarse)
    np.savez_compressed(OUT, truck=np.array(trucks), date=np.array(dates),
                        fine_idx=fv, fine_off=fo, coarse_idx=cv, coarse_off=co,
                        fine_n=FINE_N, coarse_n=COARSE_N)
    print(f"saved {OUT.name}: {len(trucks)} truck-days")


if __name__ == "__main__":
    main()
