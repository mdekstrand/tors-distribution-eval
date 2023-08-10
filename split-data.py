# -*- coding: utf-8 -*-
"""
Usage:
    split-data.py [-p PARTS] [-u USERS] [-o OUT] [-n RATES] [-R REPS] DATASET

Options:  
    -p PARTS          number of cross-folds
    -u USERS          number of test users
    -n RATES          number of test items per user [default: 5]
    -R REPS           repeatedly re-split REPS times
    -o OUT            destination directory [default: data-split]
    DATASET           name of data set to load  
"""

import sys
from docopt import docopt
from lkdemo import datasets, log
from pathlib import Path

from seedbank import init_file
import lenskit.crossfold as xf

def main(args):
    dsname = args.get('DATASET')
    partitions = args.get('-p')
    if partitions:
        partitions = int(partitions)
    nusers = args.get('-u')
    if nusers:
        nusers = int(nusers)
    output = args.get('-o')
    reps = args.get('-R')
    if reps:
        reps = int(reps)
    nrates = int(args.get('-n'))
    if partitions is None and nusers is None:
        _log.error('must specify at least 1 of -p and -u')
        sys.exit(2)

    # initialize RNG with the data set name in the seed
    init_file('params.yaml', 'split-data', dsname)

    _log.info('locating data set %s', dsname)
    data = getattr(datasets, dsname)

    _log.info('loading ratings')
    ratings = data.ratings

    path = Path(output)
    path.mkdir(exist_ok=True, parents=True)

    _log.info('writing to %s', path)
    samp = xf.SampleN(nrates)
    if partitions:
        if reps:
            _log.info('reps and partitions incompatible')
        if nusers:
            _log.info('computing %d samples of %d users', partitions, nusers)
            parts = xf.sample_users(ratings, partitions, nusers, samp)
        else:
            _log.info('computing %d partitions', partitions)
            parts = xf.partition_users(ratings, partitions, samp)
        for i, tp in enumerate(parts, 1):
            _log.info('writing test set %d', i)
            tp.test.index.name = 'index'
            tp.test.to_parquet(path / f'test-{i}.parquet', compression='brotli')
    else:
        for i in range(max(reps, 1)):
            train, test = next(xf.sample_users(ratings, 1, nusers, samp))
            fn = f'test-{i}.parquet' if reps else 'test.parquet'
            _log.info('writing test data %s', fn)
            test.index.name = 'index'
            test.to_parquet(path / fn, compression='brotli')
        

if __name__ == '__main__':
    _log = log.script(__file__)
    args = docopt(__doc__)
    main(args)
