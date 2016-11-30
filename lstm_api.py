#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
LSTM API reads an iterator of data and prepares the data for the training.



@Gini GmbH
'''

import os

import numpy as np


class FeatureValues(np.ndarray):
    """


    """
    def __new__(cls, *feature_value):
        return np.array(feature_value, dtype = float)

class Sample(np.ndarray):
    """

    """
    def __new__(cls, *feature_values):
        assert isinstance(feature_values, np.ndarray)
        assert feature_values.dtype == np.dtype('float32')
        return np.array(feature_values, dtype=np.ndarray)

class Groundtruth(np.ndarray):
    """

    """
    def __new__(cls, *targets):
        assert isinstance(targets, np.ndarray)
        assert targets.dtype == np.dtype('uint8')
        return np.array(targets, dtype = int)

class TrainingDatum(tuple):
    """
    An instance of training datum.
    It consists of a single sample sequence and corresponding groundtruth sequence.
    """

    def __new__(cls, sample, groundtruth):
        """

        :param sample:  a list of length seq_length containing n_feature feature values
        :type  sample:  np.ndarray(np.ndarray(float32))
        :param groundtruth:  class labels for the sample data
        :type  groundtruth:  np.ndarray(uint8)
        """
        return tuple.__new__(cls, (sample, groundtruth))

def target_list_to_sparse_tensor(targetList, class_mapping):
    '''make tensorflow SparseTensor from list of targets, with each element
       in the list being a list or array with the values of the target sequence
       (e.g., the integer values of a character map for an ASR target string)
       See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/ctc/ctc_loss_op_test.py
       for example of SparseTensor format'''
    indices = []
    vals = []
    for tI, target in enumerate(targetList):
        for seqI, val in enumerate(target):
            indices.append([tI, seqI])
            vals.append(class_mapping.keys()[class_mapping.values().index(val)])
    shape = [len(targetList), np.asarray(indices).max(0)[1]+1]
    print("shape = {}".format(shape))
    return (np.array(indices), np.array(vals), np.array(shape))

def get_parameters(sample_target_itr):
    """
    Extract a mapping from class indices to target labels, the max time steps and the max target seq length
    
    :type sample_target_itr: iterable over TrainingDatum

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


def load_batched_data(sample_target_itr, batch_size, n_max_time_steps, n_classes, class_mapping):
    """
    Transform the data from input iterator to learnable format.

    :type sample_target_itr: iterator
    :type batch_size: int
    :type n_max_time_steps: int
    :type n_classes: int
    :type class_mapping: mapping from class indices to target labels
    :return  batches, n_max_time_steps, i, n_classes
             batches: list((cube, target, seq_len)) with length of sample / batch size
             cube: a batch of input data as batch_size times a list of n_max_time_steps feature values
                   of the first feature, a list of n_max_time_steps feature values of the second feature, ...
             target: a batch of output data in form of a list of class indices
             seq_len: a list of input sequence lengths
    :rtype (list((np.ndarray(np.ndarray(np.ndarray(float32))), list(np.ndarray(uint8)), list(int))), int, int, int)
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
            targets.append(target_list_to_sparse_tensor(targets_in_batch, class_mapping))
            cube = list()
            seq_len = list()
            targets_in_batch = list()
        i += 1

    # don't forget the last batch
    if cube:
        cubes.append(np.array(cube))
        seq_lens.append(seq_len)
        targets.append(target_list_to_sparse_tensor(targets_in_batch, class_mapping))

    # putting it all together

    batches = list(zip(cubes, targets, seq_lens))

    return batches, n_max_time_steps, i, n_classes

