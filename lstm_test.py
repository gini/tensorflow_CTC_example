#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sample_data.sample_data as sample_data
import lstm_api as api
import numpy as np


import unittest


def recursive_type(o):
    if isinstance(o, list):
        first = o[0]
        return "list(" + recursive_type(first) + ")"
    elif isinstance(o, np.ndarray):
        first = o[0]
        return "np.ndarray(" +recursive_type(first) + ")"
    elif isinstance(o, tuple):
        return "(" + ", ".join([recursive_type(item) for item in o]) + ")"
    else:
        return type(o).__name__


class LSTMAPITest(unittest.TestCase):


    def setUp(self):

        self.INPUT_PATH = './sample_data/mfcc'      # directory of MFCC nFeatures x nFrames 2-D array .npy files
        self.TARGET_PATH = './sample_data/char_y/'  # directory of nCharacters 1-D array .npy files


    def test_get_params(self):
        itr = sample_data.createExampleIt(self.INPUT_PATH, self.TARGET_PATH)

        #print(recursive_type(next(itr)))

        class_mapping, max_time_steps, max_target_seq_len = api.get_parameters(itr)

        self.assertEqual(len(class_mapping), 24, "The number of classes in the example should be 24 characters")
        self.assertEqual(max_time_steps, 423, "Max time steps (the length of longest input sequence) should be 423.")
        self.assertEqual(max_target_seq_len, 91, "Length of the longest target sequence should be 91")

    def test_load_batch_data(self):
        itr = sample_data.createExampleIt(self.INPUT_PATH, self.TARGET_PATH)
        class_mapping, max_time_steps, max_target_seq_len = api.get_parameters(itr)
        itr = sample_data.createExampleIt(self.INPUT_PATH, self.TARGET_PATH)
        batch_size = 5
        result = api.load_batched_data(itr, batch_size, max_time_steps, len(class_mapping))
        batches, n_max_time_steps, n_samples, n_classes = result

        cube, _, _ = batches[0]

        self.assertEquals(recursive_type(result), '(list((np.ndarray(np.ndarray(np.ndarray(float32))), list(np.ndarray(uint8)), list(int))), int, int, int)')
        self.assertEqual(len(cube), batch_size)
        self.assertEqual(len(cube[0]), 26, "Number of input features should be 26.")
        self.assertEqual(len(cube[0][0]), max_time_steps)
        self.assertEqual(n_samples, 8)
        self.assertEqual(n_max_time_steps, max_time_steps)
        self.assertEqual(n_classes, len(class_mapping))





if __name__ == '__main__':
    unittest.main()