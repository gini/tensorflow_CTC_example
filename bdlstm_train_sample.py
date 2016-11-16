#!/usr/bin/env python
'''
Example of a single-layer bidirectional long short-term memory network trained with
connectionist temporal classification to predict character sequences from nFeatures x nFrames
arrays of Mel-Frequency Cepstral Coefficients.  This is test code to run on the
8-item data set in the "sample_data" directory, for those without access to TIMIT.

Author: Jon Rein
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_rnn
import numpy as np
from utils import load_batched_data

INPUT_PATH = './sample_data/mfcc' #directory of MFCC nFeatures x nFrames 2-D array .npy files
TARGET_PATH = './sample_data/char_y/' #directory of nCharacters 1-D array .npy files

####Learning Parameters
learningRate = 0.001
momentum = 0.9
nEpochs = 300
batchSize = 4

####Network Parameters
nHidden = 128

####Load data
print('Loading data')
batchedData, maxTimeSteps, totalN, nClasses = load_batched_data(INPUT_PATH, TARGET_PATH, batchSize)
nFeatures = batchedData[0][0].shape[2] #26 #12 MFCC coefficients + energy, and derivatives
print('-> loaded a total of %d samples.' % totalN)

def sparseX2matrix(indices, values, shape, default_value=0):
    r, c = shape
    idx = 0
    result = []
    for y in range(r):
        arr = []
        for x in range(c):
            if idx >= len(indices):
                arr.append(default_value)
            else:
                if indices[idx][0] == y and indices[idx][1] == x:
                    arr.append(values[idx])
                    idx = idx + 1
                else:
                    arr.append(default_value)
        result.append(arr)
    return result

def sparse2matrix(sparse_tensor, default_value=0):
    return sparseX2matrix(sparse_tensor[0], sparse_tensor[1], sparse_tensor[2], default_value)

def printTarget(chars):
    for c in chars:
        if c == 0:
            print(' ', end='')
        else:
            print(chr(64 + c), end='')
    print()

i = 0
for batchInputs, batchTargetSparse, batchSeqLengths in batchedData:
    batchTarget = sparse2matrix(batchTargetSparse)
    for batch in range(batchSize):
        print('Sample %d:' % i)

        seqLength = batchSeqLengths[batch]
        print('input sequence of length %d -> ' % seqLength, end='')
        printTarget(batchTarget[batch])
        i = i + 1
#print("totalN = ", totalN)
#print("maxTimeSteps = ", maxTimeSteps)
#print(len(batchedData)) # batches
#print(len(batchedData[0])) # input x target x seqlength
#print(batchedData[0][0].shape) # maxTimeSteps x batch samples x features
#print(batchedData[1][1]) # sparse target matrix -> indices x values x dimensions (batches x max output seqlength)
#print(batchedData[0][2])

saver = None

####Define graph
print('Defining graph (takes quite some time)')
graph = tf.Graph()
with graph.as_default():

    ####NOTE: try variable-steps inputs and dynamic bidirectional rnn, when it's implemented in tensorflow
        
    ####Graph input
    inputX = tf.placeholder(tf.float32, shape=(maxTimeSteps, batchSize, nFeatures))
    #Prep input data to fit requirements of rnn.bidirectional_rnn
    #  Reshape to 2-D tensor (nTimeSteps*batchSize, nfeatures)
    inputXrs = tf.reshape(inputX, [-1, nFeatures])
    #  Split to get a list of 'n_steps' tensors of shape (batch_size, n_hidden)
    inputList = tf.split(0, maxTimeSteps, inputXrs)
    targetIxs = tf.placeholder(tf.int64)
    targetVals = tf.placeholder(tf.int32)
    targetShape = tf.placeholder(tf.int64)
    targetY = tf.SparseTensor(targetIxs, targetVals, targetShape)
    seqLengths = tf.placeholder(tf.int32, shape=(batchSize))

    ####Weights & biases
    weightsOutH1 = tf.Variable(tf.truncated_normal([2, nHidden],
                                                   stddev=np.sqrt(2.0 / (2*nHidden))))
    biasesOutH1 = tf.Variable(tf.zeros([nHidden]))
    weightsClasses = tf.Variable(tf.truncated_normal([nHidden, nClasses],
                                                     stddev=np.sqrt(2.0 / nHidden)))
    biasesClasses = tf.Variable(tf.zeros([nClasses]))

    ####Network
    forwardH1 = rnn_cell.LSTMCell(nHidden, use_peepholes=True, state_is_tuple=True)
    backwardH1 = rnn_cell.LSTMCell(nHidden, use_peepholes=True, state_is_tuple=True)
    fbH1, _, _ = bidirectional_rnn(forwardH1, backwardH1, inputList, dtype=tf.float32,
                                       scope='BDLSTM_H1')
    fbH1rs = [tf.reshape(t, [batchSize, 2, nHidden]) for t in fbH1]
    outH1 = [tf.reduce_sum(tf.mul(t, weightsOutH1), reduction_indices=1) + biasesOutH1 for t in fbH1rs]

    logits = [tf.matmul(t, weightsClasses) + biasesClasses for t in outH1]

    ####Optimizing
    logits3d = tf.pack(logits)
    loss = tf.reduce_mean(ctc.ctc_loss(logits3d, targetY, seqLengths))
    optimizer = tf.train.MomentumOptimizer(learningRate, momentum).minimize(loss)
    
    ####Evaluating
    logitsMaxTest = tf.slice(tf.argmax(logits3d, 2), [0, 0], [seqLengths[0], 1])
    predictions = tf.to_int32(ctc.ctc_beam_search_decoder(logits3d, seqLengths)[0][0])
    errorRate = tf.reduce_sum(tf.edit_distance(predictions, targetY, normalize=False)) / \
                tf.to_float(tf.size(targetY.values))

    saver = tf.train.Saver()  # defaults to saving all variables
    ckpt = tf.train.get_checkpoint_state('./checkpoints')

    with tf.Session(graph=graph) as session:
        merged = tf.merge_all_summaries()
        writer = tf.train.SummaryWriter("/tmp/basic_new", session.graph)
        
        start = 0
        if ckpt and ckpt.model_checkpoint_path:
            p = re.compile('\./checkpoints/model\.ckpt-([0-9]+)')
            m = p.match(ckpt.model_checkpoint_path)
            start = int(m.group(1))
        if start > 0:
            # Restore variables from disk.
            saver.restore(session, "./checkpoints/model.ckpt-%d" % start)
            print("Model %d restored." % start)
        else:
            print('Initializing')
            session.run(tf.initialize_all_variables())
        for epoch in range(start, nEpochs):
            print('Epoch', epoch+1, '...')
            batchErrors = np.zeros(len(batchedData))
            batchRandIxs = np.random.permutation(len(batchedData)) #randomize batch order
            for batch, batchOrigI in enumerate(batchRandIxs):
                batchInputs, batchTargetSparse, batchSeqLengths = batchedData[batchOrigI]
                batchTargetIxs, batchTargetVals, batchTargetShape = batchTargetSparse
                feedDict = {inputX: batchInputs, targetIxs: batchTargetIxs, targetVals: batchTargetVals,
                            targetShape: batchTargetShape, seqLengths: batchSeqLengths}
                _, l, er, lmt = session.run([optimizer, loss, errorRate, logitsMaxTest], feed_dict=feedDict)
                print(np.unique(lmt)) #print unique argmax values of first sample in batch; should be blank for a while, then spit out target values
                if (batch % 1) == 0:
                    print('Minibatch', batch, '/', batchOrigI, 'loss:', l)
                    print('Minibatch', batch, '/', batchOrigI, 'error rate:', er)
                batchErrors[batch] = er*len(batchSeqLengths)
            epochErrorRate = batchErrors.sum() / totalN
            saver.save(session, 'checkpoints/model.ckpt', global_step=epoch+1)
            print('Epoch', epoch+1, 'error rate:', epochErrorRate)
        print('Learning finished')

        # Do some work with the model
        feedDict = {inputX: batchedData[0][0], seqLengths: [batchedData[0][2][0], 0, 0, 0]} 
        p = session.run(predictions, feed_dict=feedDict)
        batch = sparse2matrix(p)
        printTarget(batch[0])
