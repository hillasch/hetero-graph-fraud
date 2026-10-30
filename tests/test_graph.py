import pandas as pd

from hgfraud.data import add_entity_keys
from hgfraud.features import graph_features
from hgfraud.graph import build_graph


def toy():
    df = pd.DataFrame({
        "TransactionID": [1, 2, 3, 4],
        "TransactionDT": [10, 20, 30, 40],
        "isFraud": [1, 1, 0, 0],
        "card1": [100, 100, 200, 300],
        "addr1": [1, 1, 2, 3],
        "DeviceInfo": ["farm", "farm", None, "phone"],
        "P_emaildomain": ["x.com", "x.com", "gmail.com", "gmail.com"],
    }).set_index("TransactionID")
    return add_entity_keys(df)


def test_shared_entities_connect_transactions():
    G = build_graph(toy())
    card = [n for n in G.neighbors("tx:1") if n.startswith("card:")][0]
    assert "tx:2" in set(G.neighbors(card))


def test_no_self_label_leak():
    df = toy()
    G = build_graph(df)
    f = graph_features(df, G, train_index=df.index, prior_weight=0)
    # tx 1's only card neighbor is tx 2 (fraud) -> rate 1.0, not influenced by itself
    assert f.loc[1, "card_fraud_rate"] == 1.0
    # tx 3's card is unshared -> no other evidence -> NaN with zero prior weight
    assert pd.isna(f.loc[3, "card_fraud_rate"])
