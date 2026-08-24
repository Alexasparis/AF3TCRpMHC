#!/usr/bin/env python3
"""Self-check for explore_filters.

explore_filters computes precision/recall/FPR from raw confusion counts instead of
calling sklearn per combination. This asserts those counts agree with
sklearn.metrics on every combination it enumerates, including the degenerate
folds (no predicted positives, no true positives) where sklearn's
zero_division=0 convention applies.

Run: python test_explore_filters.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from filters import explore_filters, create_nonoverlapping_folds

TIER_MAP = {'LQ': 0, 'AQ': 1, 'MQ': 2, 'HQ': 3}


def make_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        'pdb_id': [f"p{i // 5:03d}" for i in range(n)],
        'Quality': rng.choice(['LQ', 'AQ', 'MQ', 'HQ'], size=n),
        'plddt': rng.uniform(40, 99, n),
        'iptm_mean': rng.uniform(0.1, 0.99, n),
        'pdockq': rng.uniform(0.0, 0.95, n),
        'pdockq2_a': rng.uniform(0.0, 0.95, n),
        'pdockq2_b': rng.uniform(0.0, 0.95, n),
    })
    # NaNs must binarize to False, same as the original implementation
    df.loc[df.index[:10], 'plddt'] = np.nan
    return df


def reference_row(folds, comb, ops, metrics, thresholds, neg, pos):
    """Recompute one row the slow, obvious way, using sklearn."""
    precisions, recalls, fprs = [], [], []
    for train_df, _ in folds:
        ev = train_df.copy()
        ev['tier_num'] = ev['Quality'].map(TIER_MAP)
        ev = ev[ev['tier_num'].isin(neg + pos)]
        if ev.empty:
            continue
        cols = {}
        for m in metrics:
            thr = thresholds[m]
            cols[m] = ev[m].values > thr[0 if len(thr) == 2 else 1]
        result = cols[comb[0]].copy()
        for i, op in enumerate(ops):
            result = (result & cols[comb[i + 1]]) if op == 'AND' else (result | cols[comb[i + 1]])
        y_true = ev['tier_num'].isin(pos).astype(int)
        y_pred = result.astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        FP = np.sum((y_pred == 1) & (ev['tier_num'].isin(neg)))
        TN = np.sum((y_pred == 0) & (ev['tier_num'].isin(neg)))
        fprs.append(FP / (FP + TN) if (FP + TN) > 0 else np.nan)
    return np.mean(precisions), np.mean(recalls), np.mean(fprs)


def main():
    metrics = ['plddt', 'iptm_mean', 'pdockq', 'avgpdockq2']
    thresholds = {
        'plddt': [70, 90],
        'iptm_mean': [0.6, 0.8],
        'pdockq': [0.23, 0.49, 0.8],
        'avgpdockq2': [0.23, 0.49, 0.8],
    }
    neg, pos = ['LQ'], ['AQ', 'MQ', 'HQ']

    df = make_df()
    folds = create_nonoverlapping_folds(df, n_splits=5)
    got = explore_filters(folds, metrics, thresholds, target_col='Quality',
                          tiers_negative=neg, tiers_positive=pos)

    assert len(got) == (3 ** len(metrics) - 1) // 2, len(got)

    neg_n = [TIER_MAP[t] for t in neg]
    pos_n = [TIER_MAP[t] for t in pos]
    for row in got.itertuples(index=False):
        p, r, f = reference_row(folds, row.metrics, row.operators, metrics,
                                thresholds, neg_n, pos_n)
        for name, a, b in (('precision', row.precision_mean, p),
                           ('recall', row.recall_mean, r),
                           ('fpr', row.false_positive_rate_mean, f)):
            assert np.isclose(a, b, equal_nan=True), \
                f"{name} mismatch for {row.metrics} {row.operators}: {a} != {b}"

    print(f"OK: {len(got)} combinations match sklearn exactly")


if __name__ == "__main__":
    main()
