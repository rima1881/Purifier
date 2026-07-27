#!/usr/bin/env python3
"""
Builds src/data/kitti_gps_imu.txt from a real KITTI raw sequence's oxts data.

KITTI's oxts output (from an OXTS RT3003 GPS/INS unit) is treated as ground
truth position/heading/speed. Synthetic Gaussian noise (fixed seed) is added
to position to simulate a consumer-grade GPS; the real forward-acceleration
and yaw-rate signals are used unmodified as IMU control inputs for
`gps_ins.zig`'s bicycle model.

Usage:
  1. Download the synced sequence (~616MB, includes camera/lidar data this
     script doesn't need):
       wget https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0005/2011_09_26_drive_0005_sync.zip
  2. Extract just the oxts/ folder (a few hundred KB) from it, e.g.:
       python3 -c "import zipfile; z=zipfile.ZipFile('2011_09_26_drive_0005_sync.zip'); \
         z.extractall('.', members=[n for n in z.namelist() if '/oxts/' in n])"
  3. Run this script with SRC pointed at the resulting oxts/ directory.

Output columns (tab-separated), one row per frame:
  t_rel  gps_px  gps_py  af  wu  gt_px  gt_py  gt_v  gt_yaw
"""
import math
import random
import sys

SRC = sys.argv[1] if len(sys.argv) > 1 else "2011_09_26/2011_09_26_drive_0005_sync/oxts"
OUT = sys.argv[2] if len(sys.argv) > 2 else "../data/kitti_gps_imu.txt"
GPS_NOISE_STD = 2.0  # meters, typical consumer GPS accuracy
SEED = 42


def parse_timestamp(line):
    # "2011-09-26 13:04:32.349659964"
    _date_part, time_part = line.split(" ")
    h, m, s = time_part.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def main():
    timestamps = open(f"{SRC}/timestamps.txt").read().splitlines()
    n = len(timestamps)
    t = [parse_timestamp(ts) for ts in timestamps]
    t0 = t[0]

    rows = []
    for i in range(n):
        vals = [float(x) for x in open(f"{SRC}/data/{i:010d}.txt").read().split()]
        rows.append(vals)

    lat0, lon0 = rows[0][0], rows[0][1]
    lat0r = math.radians(lat0)
    r_earth = 6378137.0

    rng = random.Random(SEED)

    out = []
    for i in range(n):
        lat, lon, _alt, _roll, _pitch, yaw, _vn, _ve, vf, vl, _vu, _ax, _ay, _az, af, _al, _au, _wx, _wy, _wz, _wf, _wl, wu = rows[i][:23]

        # Local Cartesian, equirectangular projection around the first frame.
        px = r_earth * math.radians(lon - lon0) * math.cos(lat0r)
        py = r_earth * math.radians(lat - lat0)
        v = math.hypot(vf, vl)

        gps_px = px + rng.gauss(0, GPS_NOISE_STD)
        gps_py = py + rng.gauss(0, GPS_NOISE_STD)

        t_rel = t[i] - t0
        out.append(f"{t_rel:.6f}\t{gps_px:.6f}\t{gps_py:.6f}\t{af:.6f}\t{wu:.6f}\t{px:.6f}\t{py:.6f}\t{v:.6f}\t{yaw:.6f}")

    with open(OUT, "w") as f:
        f.write("\n".join(out) + "\n")

    print(f"wrote {n} rows to {OUT}")


if __name__ == "__main__":
    main()
