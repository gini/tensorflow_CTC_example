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



