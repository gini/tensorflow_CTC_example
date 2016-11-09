#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

def load_batched_data(sample_target_itr, batch_size):
    """
    Load data ...

    :type sample_target_itr: iterator
    :type batch_size: int
    """
    # for all batches
    n_features = None
    n_max_time_steps = -1
    n_max_target_seq_len = -1

    cubes = list()
    targets = list()
    seq_lens = list()

    # per batch
    cube = list()
    targets_in_batch = list()
    seq_len = list()
    i = 0
    for sample, groundtruth in sample_target_itr:
        if n_features is None:
            n_features = len(sample[0])
        else:
            assert n_features == len(sample[0])

        if n_max_time_steps < len(sample):
            n_max_time_steps = len(sample)
        if n_max_target_seq_len < len(groundtruth):
            n_max_target_seq_len = len(groundtruth)

        cube.append(sample)
        seq_len.append(len(sample))
        targets_in_batch.append(groundtruth)

        if i % batch_size == batch_size - 1:
            cubes.append(cube)
            seq_lens.append(seq_len)
            targets.append(targets_in_batch)
            cube = list()
            seq_len = list()
            targets_in_batch = list()

    # putting it all together
    ...
    # return np.array()
