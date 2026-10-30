# Hetero-Graph Fraud

**Enhancing Fraud Detection through Heterogeneous Graph Modeling of Device Fingerprints**

Traditional fraud models score each transaction as an isolated row. Coordinated fraud rings (botnets, device farms) hide by spreading activity across many cards, IPs and devices — but they still *share* resources. This project turns the IEEE-CIS Fraud Detection data into a heterogeneous graph so those shared resources become visible, then tests whether graph features improve a tabular classifier.

## Approach

1. **Graph construction** — nodes are entities (card, device fingerprint, email domain, address, …); each transaction links the entities it uses.
2. **Topological features** — connected-component size, degree centrality, and neighborhood fraud rate ("guilt by association"), computed with NetworkX.
3. **Modeling** — XGBoost baseline on the original tabular features vs. the same model with graph features added.
4. **Evaluation** — PR-AUC and F1 on a time-based split (no future leakage).

## Project layout

```
data/raw/          IEEE-CIS CSVs (not committed — see below)
data/processed/    cached intermediate files (not committed)
notebooks/         exploration and experiments
src/hgfraud/       reusable code
  data.py          load & merge transaction + identity tables
  graph.py         build the heterogeneous graph
  features.py      extract graph features per transaction
  model.py         train / evaluate baseline vs. graph-enhanced
tests/             unit tests
reports/figures/   plots for the write-up
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Data

Download from Kaggle ([competition page](https://www.kaggle.com/c/ieee-fraud-detection/data); you must accept the rules first):

```bash
kaggle competitions download -c ieee-fraud-detection -p data/raw
unzip data/raw/ieee-fraud-detection.zip -d data/raw
```

You need `train_transaction.csv` and `train_identity.csv` in `data/raw/`.

## Quick start

```python
from hgfraud.data import load_train
from hgfraud.graph import build_graph
from hgfraud.features import graph_features
from hgfraud.model import time_split, compare

df = load_train("data/raw")
train, test = time_split(df)
G = build_graph(train)                        # graph from training period only
feats = graph_features(df, G, train_index=train.index)
results = compare(train, test, feats)
print(results)
```

## Status

- [ ] EDA on entity sharing (how often do cards/devices repeat?)
- [ ] Graph construction
- [ ] Graph feature extraction
- [ ] Baseline XGBoost
- [ ] Graph-enhanced XGBoost
- [ ] Results & write-up
