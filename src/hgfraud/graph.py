"""Build a heterogeneous graph: transactions connected to the entities they use."""

import networkx as nx
import pandas as pd

# Entity column -> node type. Edit this to add/remove entity types.
ENTITY_TYPES = {
    "card_key": "card",
    "device_key": "device",
    "addr_key": "address",
    "p_email": "email",
}


def entity_node(etype: str, value) -> str:
    return f"{etype}:{value}"


def edge_list(df: pd.DataFrame, entity_types: dict[str, str] = ENTITY_TYPES) -> pd.DataFrame:
    """Long table of (tx node, entity node, entity type) — one row per edge."""
    frames = []
    tx_nodes = "tx:" + df.index.astype(str)
    for col, etype in entity_types.items():
        if col not in df.columns:
            continue
        mask = df[col].notna().to_numpy()
        frames.append(pd.DataFrame({
            "tx": tx_nodes[mask],
            "entity": etype + ":" + df.loc[mask, col].astype(str).to_numpy(),
            "etype": etype,
        }))
    return pd.concat(frames, ignore_index=True)


def build_graph(df: pd.DataFrame, entity_types: dict[str, str] = ENTITY_TYPES) -> nx.Graph:
    """Bipartite-style heterogeneous graph.

    Transaction nodes are ``tx:<id>`` with attribute ``is_fraud``.
    Entity nodes are ``<type>:<value>`` with attribute ``ntype``. Each
    transaction gets an edge to every non-missing entity it uses.
    Expects keys from ``data.add_entity_keys``.
    """
    G = nx.Graph()
    labels = df["isFraud"] if "isFraud" in df.columns else pd.Series(None, index=df.index)
    G.add_nodes_from(
        (f"tx:{i}", {"ntype": "transaction", "is_fraud": None if pd.isna(y) else int(y)})
        for i, y in labels.items()
    )
    edges = edge_list(df, entity_types)
    G.add_nodes_from((e, {"ntype": t}) for e, t in
                     edges.drop_duplicates("entity")[["entity", "etype"]].itertuples(index=False))
    G.add_edges_from((a, b, {"etype": t}) for a, b, t in edges.itertuples(index=False))
    return G


def graph_summary(G: nx.Graph) -> pd.Series:
    types = pd.Series(nx.get_node_attributes(G, "ntype")).value_counts()
    types["edges"] = G.number_of_edges()
    types["components"] = nx.number_connected_components(G)
    return types
