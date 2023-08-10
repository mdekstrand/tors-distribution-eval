# -*- coding: utf-8 -*-
"""
Usage:
    compute-metrics.py [options] DATA ALGO

Options:  
    -o FILE           output to FILE.
    DATA              name of data set to load.
    ALGO              name of the algorithm to scan.
"""

import sys
import re
from pathlib import Path
from docopt import docopt
from lkdemo import log
from pathlib import Path
from joblib import Parallel, delayed

import pandas as pd

from lenskit import topn
from lkdemo.metrics import rbp

run_dir = Path('runs')

_fn_re = re.compile(r'recs-(\d+)\.parquet')


def eval_run(data, runf: Path):
    test_dir = Path('data-split') / data
    m = _fn_re.match(runf.name)
    part = m.group(1)
    
    recs = pd.read_parquet(runf)
    test = pd.read_parquet(test_dir / f'test-{part}.parquet')
    
    rla = topn.RecListAnalysis()
    rla.add_metric(rbp, k=1000)
    rla.add_metric(topn.ndcg, k=1000)
    rla.add_metric(topn.recip_rank, k=1000)
    rla.add_metric(topn.hit, k=1000)
    rla.add_metric(topn.hit, name='hit10', k=10)
    rla.add_metric(topn.hit, name='hit20', k=20)
    
    vals = rla.compute(recs, test, include_missing=True)
    return (part, vals)


def main(args):
    run_dir = Path('runs')

    data = args['DATA']
    algo = args['ALGO']
    outf = args.get('-o')
    if not outf:
        outf = run_dir / f'{data}-{algo}-scores.parquet'

    rd = run_dir / f'{data}-{algo}'
    run_files = list(rd.glob(f'recs-*.parquet'))
    _log.info('scanning %d run files for %s-%s', len(run_files), data, algo)

    with Parallel(n_jobs=-1, verbose=10) as loop:
        res = dict(loop(delayed(eval_run)(data, path) for path in run_files))
    
    _log.info('finished, compiling')
    scores = pd.concat(res, names=['run'])
    _log.info('scores:\n%s', scores)
    _log.info('saving to %s', outf)
    scores.to_parquet(outf, compression='brotli')


if __name__ == '__main__':
    _log = log.script(__file__)
    args = docopt(__doc__)
    main(args)
