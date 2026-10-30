"""Load and merge the IEEE-CIS transaction and identity tables."""

from pathlib import Path

import pandas as pd

# Columns needed to build entity keys (graph nodes).
ENTITY_SOURCE_COLS = {
    "card_key": ["card1", "card2", "card3", "card5", "addr1"],
    "device_key": ["DeviceType", "DeviceInfo", "id_30", "id_31", "id_33"],
    "addr_key": ["addr1", "addr2"],
    "p_email": ["P_emaildomain"],
    "r_email": ["R_emaildomain"],
}
BASE_COLS = ["TransactionID", "TransactionDT", "TransactionAmt", "ProductCD", "isFraud"]


def load_train(raw_dir: str | Path = "data/raw", nrows: int | None = None,
               usecols: list[str] | None = None) -> pd.DataFrame:
    """Return train_transaction left-joined with train_identity, indexed by TransactionID.

    ``usecols`` limits loaded columns (identity columns are filtered too), which
    keeps memory low for exploration. ``nrows`` reads only the first N transactions.
    """
    raw_dir = Path(raw_dir)
    wanted = None if usecols is None else set(usecols) | {"TransactionID"}

    tx = pd.read_csv(raw_dir / "train_transaction.csv", nrows=nrows,
                     usecols=(lambda c: c in wanted) if wanted else None)
    ident = pd.read_csv(raw_dir / "train_identity.csv")
    # test_identity uses "id-01" style names; normalise to "id_01".
    ident.columns = [c.replace("-", "_") for c in ident.columns]
    if wanted:
        ident = ident[[c for c in ident.columns if c in wanted]]

    df = tx.merge(ident, on="TransactionID", how="left")
    return df.set_index("TransactionID").sort_values("TransactionDT")


def entity_columns() -> list[str]:
    """All raw columns needed for ``add_entity_keys`` plus the basics."""
    cols = list(BASE_COLS)
    for src in ENTITY_SOURCE_COLS.values():
        cols += [c for c in src if c not in cols]
    return cols


def add_entity_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Add composite entity keys used as graph nodes.

    card1 alone is not a unique card, so card fields are combined with the
    billing address to approximate a real card identity. The device
    fingerprint combines device info with OS / browser / screen fields.
    A key is missing (NA) when all its source fields are missing.
    """
    out = df.copy()
    for key, cols in ENTITY_SOURCE_COLS.items():
        present = [c for c in cols if c in out.columns]
        if not present:
            out[key] = pd.NA
            continue
        parts = out[present].astype("string")
        joined = parts.iloc[:, 0].fillna("NA")
        for c in present[1:]:
            joined = joined + "|" + parts[c].fillna("NA")
        out[key] = joined.where(parts.notna().any(axis=1))
    return out
