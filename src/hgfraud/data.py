"""Load and merge the IEEE-CIS transaction and identity tables."""

from pathlib import Path

import pandas as pd


def load_train(raw_dir: str | Path = "data/raw", nrows: int | None = None) -> pd.DataFrame:
    """Return train_transaction left-joined with train_identity, indexed by TransactionID."""
    raw_dir = Path(raw_dir)
    tx = pd.read_csv(raw_dir / "train_transaction.csv", nrows=nrows)
    ident = pd.read_csv(raw_dir / "train_identity.csv")
    # Kaggle's test_identity uses "id-01" style names; normalise to "id_01".
    ident.columns = [c.replace("-", "_") for c in ident.columns]
    df = tx.merge(ident, on="TransactionID", how="left")
    return df.set_index("TransactionID").sort_values("TransactionDT")


def add_entity_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Add composite entity keys used as graph nodes.

    card1 alone is not a unique card, so we combine card fields with the
    billing address to approximate a real card identity. Device fingerprint
    combines device info with browser / OS / screen identity fields.
    """
    out = df.copy()

    def combo(cols):
        present = [c for c in cols if c in out.columns]
        parts = out[present].astype("string")
        key = parts.fillna("NA").agg("|".join, axis=1)
        # A key that is entirely missing is not a real shared entity.
        return key.where(parts.notna().any(axis=1))

    out["card_key"] = combo(["card1", "card2", "card3", "card5", "addr1"])
    out["device_key"] = combo(["DeviceType", "DeviceInfo", "id_30", "id_31", "id_33"])
    out["addr_key"] = combo(["addr1", "addr2"])
    out["p_email"] = out.get("P_emaildomain")
    out["r_email"] = out.get("R_emaildomain")
    return out
