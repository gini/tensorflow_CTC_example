#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import numpy as np

INPUT_PATH = './sample_data/mfcc'  # directory of MFCC nFeatures x nFrames 2-D array .npy files
TARGET_PATH = './sample_data/char_y/'  # directory of nCharacters 1-D array .npy files


def createExampleIt(specPath=INPUT_PATH, targetPath=TARGET_PATH):
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


def get_parameters(sample_target_itr):
    """
    Extract a mapping from class indices to target labels, the max time steps and the max target seq length
    
    :type sample_target_itr: iterator over samples and groundtruth
    """

    labels = set()
    max_time_steps = 0
    max_target_seq_len = 0

    for sample, groundtruth in sample_target_itr:

        current_time_steps = len(sample[0])
        if current_time_steps > max_time_steps:
            max_time_steps = current_time_steps

        current_target_seq_len = len(groundtruth)
        if current_target_seq_len > max_target_seq_len:
            max_target_seq_len = current_target_seq_len

        for t in groundtruth:
            labels.add(t)

    class_mapping = dict()
    i = 0
    for l in labels:
        class_mapping[i] = l
        i += 1

    return class_mapping, max_time_steps, max_target_seq_len


def load_batched_data(sample_target_itr, batch_size, n_max_time_steps, n_classes):
    """
    Load data ...

    :type sample_target_itr: iterator
    :type batch_size: int
    :type n_max_time_steps: int
    :type n_classes: int
    """
    # for all batches
    n_features = None

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
            n_features = len(sample)
        else:
            assert n_features == len(sample)

        sample_len = len(sample[0])
        padded_sample = np.pad(np.array(sample), ((0, 0), (0, n_max_time_steps - sample_len)), 'constant',
                               constant_values=0)
        cube.append(padded_sample)
        seq_len.append(sample_len)
        targets_in_batch.append(groundtruth)

        if i % batch_size == batch_size - 1:
            cubes.append(np.array(cube))
            seq_lens.append(seq_len)
            targets.append(targets_in_batch)
            cube = list()
            seq_len = list()
            targets_in_batch = list()
        i += 1

    # don't forget the last batch
    if cube:
        cubes.append(np.array(cube))
        seq_lens.append(seq_len)
        targets.append(targets_in_batch)

    # putting it all together

    batches = list(zip(cubes, targets, seq_lens))

    return batches, n_max_time_steps, i, n_classes

if __name__ == '__main__':
    itr = createExampleIt()
    class_mapping, max_time_steps, max_target_seq_len = get_parameters(itr)
    print(u"Num Classes: {0:d}\tNum max time steps: {1:d}\tNum max target seq len: {2:d}\t".format(len(class_mapping),
                                                                                                   max_time_steps,
                                                                                                   max_target_seq_len))
    itr2 = createExampleIt()
    batches, n_max_time_steps, i, n_classes = load_batched_data(itr2, 5, max_time_steps, len(class_mapping))
    print("batches {}".format(batches[0][0][0]))
    print("max time steps {}".format(n_max_time_steps))
    print("total samples  {}".format(i))
    print("n classes  {}".format(n_classes))
