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


def build_graph(df: pd.DataFrame, entity_types: dict[str, str] = ENTITY_TYPES) -> nx.Graph:
    """Bipartite-style heterogeneous graph.

    Transaction nodes are ``tx:<id>`` with attribute ``is_fraud``.
    Entity nodes are ``<type>:<value>``. Each transaction gets an edge to
    every non-missing entity it uses. Expects keys from ``data.add_entity_keys``.
    """
    G = nx.Graph()
    has_label = "isFraud" in df.columns

    for tx_id, row in df.iterrows():
        tx = f"tx:{tx_id}"
        G.add_node(tx, ntype="transaction",
                   is_fraud=int(row["isFraud"]) if has_label else None)
        for col, etype in entity_types.items():
            val = row.get(col)
            if pd.notna(val):
                ent = entity_node(etype, val)
                if ent not in G:
                    G.add_node(ent, ntype=etype)
                G.add_edge(tx, ent, etype=etype)
    return G


def graph_summary(G: nx.Graph) -> pd.Series:
    types = pd.Series(nx.get_node_attributes(G, "ntype")).value_counts()
    types["edges"] = G.number_of_edges()
    types["components"] = nx.number_connected_components(G)
    return types
