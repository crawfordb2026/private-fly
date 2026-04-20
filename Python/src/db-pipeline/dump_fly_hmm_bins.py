#!/usr/bin/env python3
"""
Export 5-min MT_sum / Pn_var bins for one fly, merged with hmm_states, to CSV.
Uses the same binning rules as 2.5-hmm.py (complete minutes only, n_obs == 5).

PowerShell — set password on its own line (bash heredocs do not work in PowerShell):

    $env:DB_PASSWORD = "postgres"
    python dump_fly_hmm_bins.py M13_09_25_25_Ch27
    python dump_fly_hmm_bins.py M13_09_25_25_Ch27 1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from sqlalchemy import create_engine

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))
from config import DB_CONFIG, DATABASE_URL  # noqa: E402

BIN_MINUTES = 5


def load_binned_mt_pn_one_fly(experiment_id: int, fly_id: str) -> pd.DataFrame:
    """Same logic as 2.5-hmm.load_binned_mt_pn_and_metadata, one fly only (fast)."""
    engine = create_engine(DATABASE_URL)

    df = pd.read_sql(
        """
        SELECT
            r.fly_id,
            r.datetime,
            r.reading_type,
            r.value,
            f.genotype,
            f.sex,
            f.treatment
        FROM readings r
        JOIN flies f
          ON f.fly_id = r.fly_id AND f.experiment_id = r.experiment_id
        WHERE r.experiment_id = %(eid)s
          AND r.fly_id = %(fid)s
          AND r.reading_type IN ('MT', 'Pn')
        ORDER BY r.datetime
        """,
        engine,
        params={"eid": experiment_id, "fid": fly_id},
    )
    engine.dispose()

    if df.empty:
        return df

    df["datetime"] = pd.to_datetime(df["datetime"])

    wide = (
        df.pivot_table(
            index=["fly_id", "datetime", "genotype", "sex", "treatment"],
            columns="reading_type",
            values="value",
            aggfunc="mean",
        )
        .reset_index()
    )

    wide = wide.dropna(subset=["MT", "Pn"])
    wide["bin_start"] = wide["datetime"].dt.floor(f"{BIN_MINUTES}min")

    def pn_var_ddof1(x: pd.Series) -> float:
        return float(np.var(x.to_numpy(dtype=float), ddof=1))

    binned = (
        wide.groupby(["fly_id", "bin_start", "genotype", "sex", "treatment"], sort=False)
        .agg(
            n_obs=("datetime", "size"),
            MT_sum=("MT", "sum"),
            Pn_var=("Pn", pn_var_ddof1),
        )
        .reset_index()
    )

    binned = binned[binned["n_obs"] == BIN_MINUTES].copy()
    binned["MT_sum"] = binned["MT_sum"].astype(float)
    binned["Pn_var"] = binned["Pn_var"].astype(float).fillna(0.0)
    binned = binned.sort_values(["fly_id", "bin_start"]).reset_index(drop=True)
    return binned[
        ["fly_id", "bin_start", "genotype", "sex", "treatment", "MT_sum", "Pn_var"]
    ]


def _latest_experiment_id() -> int:
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT experiment_id FROM experiments ORDER BY created_at DESC LIMIT 1"
        )
        row = cur.fetchone()
        if not row:
            raise RuntimeError("No rows in experiments.")
        return int(row[0])
    finally:
        conn.close()


def main() -> None:
    p = argparse.ArgumentParser(description="Dump binned MT/Pn + HMM states for one fly.")
    p.add_argument("fly_id", help="e.g. M13_09_25_25_Ch27")
    p.add_argument(
        "experiment_id",
        nargs="?",
        type=int,
        default=None,
        help="experiment_id (default: latest by created_at)",
    )
    p.add_argument(
        "-o",
        "--output",
        default=None,
        help="CSV path (default: <fly_id>_binned_mt_pn_with_states.csv)",
    )
    args = p.parse_args()

    eid = args.experiment_id if args.experiment_id is not None else _latest_experiment_id()

    print(f"experiment_id={eid}  fly_id={args.fly_id}")
    fly = load_binned_mt_pn_one_fly(eid, args.fly_id)
    if fly.empty:
        raise SystemExit("No binned data for this fly/experiment (check fly_id and readings).")

    fly = fly.sort_values("bin_start").reset_index(drop=True)
    fly["bin_index"] = range(len(fly))

    engine = create_engine(DATABASE_URL)
    try:
        states = pd.read_sql(
            """
            SELECT bin_index, state, state_label
            FROM hmm_states
            WHERE experiment_id = %(eid)s AND fly_id = %(fid)s
            ORDER BY bin_index
            """,
            engine,
            params={"eid": eid, "fid": args.fly_id},
        )
    finally:
        engine.dispose()

    out = fly.merge(states, on="bin_index", how="left")
    out_path = (
        args.output
        or f"{args.fly_id.replace('/', '-')}_binned_mt_pn_with_states.csv"
    )
    out.to_csv(out_path, index=False)

    print(f"rows: {len(out)}")
    print(f"csv: {os.path.abspath(out_path)}")
    print("\nHead:")
    print(out.head(12).to_string(index=False))
    print("\nMT_sum:")
    print(out["MT_sum"].describe().to_string())
    print("\nPn_var:")
    print(out["Pn_var"].describe().to_string())
    print("\nstate_label:")
    print(out["state_label"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
