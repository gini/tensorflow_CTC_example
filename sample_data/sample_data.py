#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import numpy as np
from PIL import Image

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


INPUT_PATH = './smart_doc_data/test24082016/'
TARGET_PATH = './smart_doc_data/test24082016/'


def z_score_normalize(array):
    mean = np.mean(array)
    stddev = np.std(array)
    array -= mean
    array /= stddev
    return array


def create_smart_doc_it(spec_path=INPUT_PATH, target_path=TARGET_PATH, norm_height=48):
    """
    Iterator over the SmartDoc data set

    :type spec_path: path to directory containing sample image directories
    :type target_path: path to directory containing groundtruth text files
    :type norm_height: height in pixels to normalize
    """
    target_files = os.listdir(target_path)
    for target_file in target_files:
        if target_file.endswith(".txt"):
            gt_lines = list()
            with open(os.path.join(target_path, target_file), encoding='utf-8') as f:
                gt_lines = f.readlines()
            img_name = target_file[:-4]
            for line_img in os.listdir(os.path.join(spec_path, img_name)):
                # line_img has format <img_name>_<line_number 2 digits>.jpg
                line_number = int(line_img[-6:-4])
                if line_number >= len(gt_lines):
                    print(line_img)
                    # Image.open(os.path.join(specPath, img_name, line_img)).show()
                    continue
                target = np.array([np.uint8(ord(c)) for c in gt_lines[line_number].strip()])  # skip newline char
                with Image.open(os.path.join(spec_path, img_name, line_img)) as img:
                    width, height = img.size
                    size_normalized_img = img.resize((int((width * norm_height) / height + .5), norm_height), Image.LANCZOS)
                    sample = z_score_normalize(np.array(size_normalized_img).astype(np.float32))
                    yield (sample, target)
