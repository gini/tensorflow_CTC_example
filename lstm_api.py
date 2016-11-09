#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

import os
INPUT_PATH = './sample_data/mfcc' #directory of MFCC nFeatures x nFrames 2-D array .npy files
TARGET_PATH = './sample_data/char_y/' #directory of nCharacters 1-D array .npy files

def createExampleIt(specPath = INPUT_PATH, targetPath = TARGET_PATH):
    """
    Iterator over the example data

    :type specPath: path to directory containing sample .npy files
    :type targetPath: path to directory containing target .npy files
    """
    sample_files = os.listdir(specPath)
    target_files = os.listdir(targetPath)
    assert len(sample_files) == len(target_files)
    for i in range(len(sample_files)):
        yield (np.load(os.path.join(specPath, sample_files[i])), np.load(os.path.join(targetPath, target_files[i])))


def loadClasses2targetLabel(sample_target_itr):
    """
    Extract a mapping from class indices to target labels
    
    :type sample_target_itr: iterator over samples and groundtruth
    """
    labels = set()
    for _, groundtruth in sample_target_itr:
        for t in groundtruth:
             labels.add(t)
    result = dict()
    i = 0
    for l in labels:
         result[i] = l
         i += 1
    return result

#print loadClasses2targetLabel(createExampleIt())

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
    #...
    # return np.array()
