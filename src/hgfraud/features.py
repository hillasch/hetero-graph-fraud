"""Per-transaction graph features.

Leakage note: neighborhood fraud rates must only use labels from the
training period, and a transaction must never see its own label.
"""

import networkx as nx
import pandas as pd

from .graph import ENTITY_TYPES, entity_node


def entity_stats(G: nx.Graph, train_ids: set) -> pd.DataFrame:
    """For each entity node: degree, and fraud count/total among *training* transactions."""
    rows = {}
    for node, data in G.nodes(data=True):
        if data.get("ntype") == "transaction":
            continue
        n_fraud = n_total = 0
        for nb in G.neighbors(node):
            tx_id = nb.split(":", 1)[1]
            if tx_id in train_ids:
                n_total += 1
                n_fraud += G.nodes[nb]["is_fraud"] or 0
        rows[node] = (G.degree(node), n_fraud, n_total)
    return pd.DataFrame.from_dict(rows, orient="index", columns=["degree", "n_fraud", "n_total"])


def graph_features(df: pd.DataFrame, G: nx.Graph, train_index,
                   entity_types: dict[str, str] = ENTITY_TYPES,
                   prior_weight: float = 10.0) -> pd.DataFrame:
    """Return a DataFrame (indexed like ``df``) of graph features.

    Features per entity type:
      ``<type>_degree``     how many transactions share this entity
      ``<type>_fraud_rate`` smoothed fraud rate of the entity's *other* training transactions
    Plus ``component_size`` of the transaction's connected component.
    """
    train_ids = {str(i) for i in train_index}
    stats = entity_stats(G, train_ids)
    labels = df["isFraud"] if "isFraud" in df.columns else pd.Series(0, index=df.index)
    prior = labels.loc[train_index].mean()

    comp_size = {}
    for comp in nx.connected_components(G):
        for n in comp:
            comp_size[n] = len(comp)

    out = pd.DataFrame(index=df.index)
    for col, etype in entity_types.items():
        keys = df[col].map(lambda v: entity_node(etype, v) if pd.notna(v) else None)
        s = stats.reindex(keys.values)
        s.index = df.index
        # leave-one-out: drop the transaction's own label if it is in training
        in_train = df.index.astype(str).isin(train_ids)
        own = labels.where(in_train, 0).fillna(0)
        n_fraud = s["n_fraud"] - own
        n_total = s["n_total"] - in_train.astype(int)
        out[f"{etype}_degree"] = s["degree"]
        out[f"{etype}_fraud_rate"] = (n_fraud + prior_weight * prior) / (n_total + prior_weight)

    out["component_size"] = [comp_size.get(f"tx:{i}", 1) for i in df.index]
    return out
