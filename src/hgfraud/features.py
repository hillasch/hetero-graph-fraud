"""Per-transaction graph features.

Leakage note: neighborhood fraud rates only use labels from the training
period, and a transaction never sees its own label (leave-one-out).
"""

import networkx as nx
import pandas as pd

from .graph import ENTITY_TYPES, edge_list


def graph_features(df: pd.DataFrame, G: nx.Graph | None, train_index,
                   entity_types: dict[str, str] = ENTITY_TYPES,
                   prior_weight: float = 10.0) -> pd.DataFrame:
    """Return a DataFrame (indexed like ``df``) of graph features.

    Per entity type:
      ``<type>_degree``     number of transactions (in ``df``) sharing this entity
      ``<type>_fraud_rate`` smoothed fraud rate of the entity's *other* training
                            transactions, shrunk toward the global rate by ``prior_weight``
    Plus ``component_size``: number of transactions in the same connected component of ``G``
    (skipped if ``G`` is None).
    """
    train_mask = df.index.isin(train_index)
    labels = df["isFraud"] if "isFraud" in df.columns else pd.Series(0, index=df.index)
    prior = labels[train_mask].mean()

    in_train = pd.Series(train_mask.astype(int), index=df.index)
    own_fraud = labels.where(train_mask, 0).fillna(0)

    out = pd.DataFrame(index=df.index)
    for col, etype in entity_types.items():
        if col not in df.columns:
            continue
        key = df[col]
        tr = pd.DataFrame({"key": key, "y": own_fraud, "t": in_train})
        grp = tr.groupby("key", dropna=True)
        degree = key.map(grp.size())
        n_total = key.map(grp["t"].sum()) - in_train
        n_fraud = key.map(grp["y"].sum()) - own_fraud
        out[f"{etype}_degree"] = degree
        out[f"{etype}_fraud_rate"] = (n_fraud + prior_weight * prior) / (n_total + prior_weight)

    if G is not None:
        comp_size = {}
        for comp in nx.connected_components(G):
            size = sum(1 for n in comp if n.startswith("tx:"))  # transactions only
            for n in comp:
                comp_size[n] = size
        out["component_size"] = [comp_size.get(f"tx:{i}", 1) for i in df.index]
    return out


__all__ = ["graph_features", "edge_list"]
