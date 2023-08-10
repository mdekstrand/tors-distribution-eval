import numpy as np
import pandas as pd

from lenskit.metrics.topn import bulk_impl

PATIENCE = 0.8


def discount(ranks, patience=PATIENCE):
    return patience ** ranks


def test_weight(n, patience=PATIENCE):
    return (1 - patience ** n) / (n * (1 - patience))


def rbp_max(ngood, patience=PATIENCE):
    max = np.sum(patience ** np.arange(1, ngood+1))
    max *= (1 - patience)
    return max


def rbp(recs, truth, k=None, patience=PATIENCE):
    if k is not None:
        recs = recs.iloc[:k]

    nrel = len(truth)
    if nrel == 0:
        return None
    
    good = recs['item'].isin(truth.index)
    ranks = recs['rank'][good]
    disc = discount(ranks, patience)
    return np.sum(disc) * (1 - patience)


@bulk_impl(rbp)
def _bulk_rbp(recs, truth, k=None, patience=PATIENCE):
    if k is not None:
        recs = recs[recs['rank'] <= k]

    good = recs.join(truth, on=['LKTruthID', 'item'], how='inner')
    good['rbp_disc'] = discount(good['rank'], patience)
    scores = good.groupby('LKRecID')['rbp_disc'].sum()
    scores *= (1 - patience)
    return scores
