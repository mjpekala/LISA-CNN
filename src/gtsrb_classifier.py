#!/bin/env python

""" Code for training and then attacking a simple street sign detector.
"""


import sys, os
import random
import math
from PIL import Image
import pdb

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import matplotlib as mpl
mpl.use('Agg')  # for matplotlib with no display / X server
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tensorflow.python.ops.image_ops import per_image_standardization

import keras

from cleverhans.attacks import FastGradientMethod, BasicIterativeMethod, SaliencyMapMethod, VirtualAdversarialMethod, CarliniWagnerL2, ElasticNetMethod
from cleverhans.utils_keras import cnn_model, KerasModelWrapper
from cleverhans.utils_tf import model_train, model_eval, batch_eval

from utils import *
import subimage
import gtsrb



def load_gtsrb_data(image_dir, output_dir):
    """
    """
    cache_fn = os.path.join(output_dir, 'gtsrb_data.npz')

    # Create the dataset if it does not already exist.
    # Note that we create it once, up front, so that we
    # can ensure a consistent train/test split across all experiments.
    if not os.path.exists(cache_fn):
        print('[preprocess_data]: Extracting sign images ...please wait...')

        si = gtsrb.parse_train(image_dir)
        train_idx, test_idx = gtsrb.default_train_test_split(si)
        print(si.describe(train_idx))
        print(si.describe(test_idx))

        #
        # signs *without* any extra context
        #
        x_train, y_train = si.get_subimages(train_idx, (32,32), 0.0)
        x_train = np.array(x_train) # list -> tensor
        y_train = np.array(y_train).astype(np.int32)

        x_test, y_test = si.get_subimages(test_idx, (32,32), 0.0)
        x_test = np.array(x_test) # list -> tensor
        y_test = np.array(y_test).astype(np.int32)

        # (optional) rescale
        if True:
            x_train = x_train.astype(np.float32) / 255.
            x_test = x_test.astype(np.float32) / 255.

        # save for quicker reload next time
        makedirs_if_needed(os.path.dirname(cache_fn))
        np.savez(cache_fn, train_idx=train_idx,  
                           test_idx=test_idx, 
                           x_train=x_train, 
                           y_train=y_train, 
                           x_test=x_test, 
                           y_test=y_test)

    f = np.load(cache_fn)
    x_train = f['x_train']
    y_train = f['y_train']
    x_test = f['x_test']
    y_test = f['y_test']

    for string, data in zip(['train data', 'train labels', 'test data', 'test labels'], [x_train, y_train, x_test, y_test]):
        print('[load_lisa_data]: ', string, data.shape, data.dtype, np.min(data), np.max(data))

    return x_train, y_train, x_test, y_test







def make_cnn(sess, batch_size, dim, num_classes=43):
    """  Creates a simple sign classification network.
    
    Note that the network produced by cnn_model() is fairly weak.
    For example, on CIFAR-10 it gets 60-something percent accuracy,
    which substantially below state-of-the-art.

    Note: it is not required here that dim be the same as the
    CNN input spatial dimensions.  In these cases, the
    caller is responsible for resizing x to make it
    compatible with the model (e.g. via random crops).
    """
    num_channels=1  # assuming grayscale for now
    x = tf.placeholder(tf.float32, shape=(batch_size, dim, dim, num_channels))
    y = tf.placeholder(tf.float32, shape=(batch_size, num_classes))

    # XXX: set layer naming convention explicitly? 
    #      Otherwise, names depend upon when model was created...
    model = simple_cnn_model(input_shape=(dim,dim,num_channels), nb_classes=num_classes)

    return model, x, y



def train_cnn(sess, data, cnn_weight_file, batch_size=128):
    """ Trains a sign classifier.
    """
    X_train, Y_train, X_test, Y_test = data

    model, x, y = make_cnn(sess, batch_size, X_train.shape[1])

    # construct an explicit predictions variable
    model_output = model(x)

    def evaluate():
        # Evaluate accuracy of the model on clean test examples.
        preds = run_in_batches(sess, x, y, model_output, X_test, Y_test, batch_size)
        print('test accuracy: ', calc_acc(Y_test, preds))

    # Note: model_train() will add some new (Adam-related) variables to the graph
    train_params = {
        'nb_epochs': 30,
        'batch_size': batch_size,
        'learning_rate': 0.0001, 
    }
    model_train(sess, x, y, model_output, X_train, Y_train, evaluate=evaluate, args=train_params)

    saver = tf.train.Saver()
    save_path = saver.save(sess, cnn_weight_file)
    print("[train_cnn]: model was saved to " +  cnn_weight_file)




def attack_cnn(sess, data, cnn_weight_file, out_dir, y_target=None, batch_size=128):
    """ Generates AE for the LISA-CNN.
        Assumes you have already run train_lisa_cnn() to train the network.
    """
    epsilon_map = {np.inf : [.02, .05, .075, .1, .15, .2],   # assumes values in [0,1]
                        1 :      [.1, 1, 10], 
                        2 :      [.1, 1, 10]}

    #--------------------------------------------------
    # data prep
    #--------------------------------------------------
    X_train, Y_train, X_test, Y_test = data

    # Create one-hot target labels (needed for targeted attacks only)
    if y_target is not None:
        Y_target_OB = categorical_matrix(y_target, batch_size, Y_test.shape[1])
        Y_target = categorical_matrix(y_target, Y_test.shape[0], Y_test.shape[1])
    else:
        Y_target_OB = None
        Y_target = None

    # bound the perturbation
    c_max = np.max(X_test)
    assert(c_max <= 1.0) # assuming this for now

    #--------------------------------------------------
    # Initialize model that we will attack
    #--------------------------------------------------
    model, x_tf, y_tf = make_cnn(sess, batch_size, X_train.shape[1])
    model_CH = KerasModelWrapper(model) # to make CH happy
    model_output = model(x_tf)

    saver = tf.train.Saver()
    saver.restore(sess, cnn_weight_file)

    #--------------------------------------------------
    # Performance on clean data
    # (try this before attacking)
    #--------------------------------------------------
    predictions = run_in_batches(sess, x_tf, y_tf, model_output, X_test, Y_test, batch_size)
    acc_clean = calc_acc(Y_test, predictions)
    print('[info]: accuracy on clean test data: %0.2f' % acc_clean)
    print(confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(predictions, axis=1)))

    save_images_and_estimates(X_test, Y_test, predictions, os.path.join(out_dir, 'Images', 'Original'))


    #--------------------------------------------------
    # Fast Gradient Attack
    #--------------------------------------------------
    # symbolic representation of attack
    attack = FastGradientMethod(model_CH, sess=sess)
    acc_fgm = {}
    acc_tgt_fgm = {}

    for ord in [np.inf, 1, 2]:
        epsilon_values = epsilon_map[ord]
        acc_fgm[ord] = []
        acc_tgt_fgm[ord] = []

        for idx, epsilon in enumerate(epsilon_values):
            desc = 'FGM-ell%s-%0.3f' % (ord, epsilon)

            x_adv_tf = attack.generate(x_tf, eps=epsilon, y_target=Y_target_OB, clip_min=0.0, clip_max=c_max, ord=ord)

            if Y_target is not None:
                X_adv = run_in_batches(sess, x_tf, y_tf, x_adv_tf, X_test, Y_target, batch_size)
            else:
                X_adv = run_in_batches(sess, x_tf, y_tf, x_adv_tf, X_test, Y_test, batch_size)

            #
            # Evaluate the AE. 
            # Currently using the same model we originally attacked.
            #
            model_eval = model
            preds_tf = model_eval(x_tf)
            preds = run_in_batches(sess, x_tf, y_tf, preds_tf, X_adv, Y_test, batch_size)
            acc, acc_tgt = analyze_ae(X_test, X_adv, Y_test, preds, desc, y_target)

            save_images_and_estimates(X_adv, Y_test, preds, os.path.join(out_dir, 'Images', desc))
            #save_images_and_estimates(X_test - X_adv, Y_test, os.path.join(out_dir, 'Images', desc))
            acc_fgm[ord].append(acc)
            acc_tgt_fgm[ord].append(acc_tgt)


    #--------------------------------------------------
    # Iterative attack
    #--------------------------------------------------
    attack = BasicIterativeMethod(model_CH, sess=sess)
    acc_ifgm = {}
    acc_tgt_ifgm = {}

    for ord in [np.inf, 1, 2]:
        epsilon_values = epsilon_map[ord]
        acc_ifgm[ord] = []
        acc_tgt_ifgm[ord] = []

        for idx, epsilon in enumerate(epsilon_values):
            desc = 'I-FGM-ell%s-%0.3f' % (ord, epsilon)

            x_adv_tf = attack.generate(x_tf, eps=epsilon, 
                                         eps_iter=epsilon/4., 
                                         nb_iter=100,
                                         y_target=Y_target_OB, 
                                         clip_min=0.0,
                                         clip_max=c_max)

            #
            # Run the attack (targeted or untargeted)
            # on the test data.
            #
            if Y_target is not None:
                X_adv = run_in_batches(sess, x_tf, y_tf, x_adv_tf, X_test, Y_target, batch_size)
            else:
                X_adv = run_in_batches(sess, x_tf, y_tf, x_adv_tf, X_test, Y_test, batch_size)

            #
            # Evaluate the AE. 
            # Currently using the same model we originally attacked.
            #
            model_eval = model
            preds_tf = model_eval(x_tf)
            preds = run_in_batches(sess, x_tf, y_tf, preds_tf, X_adv, Y_test, batch_size)
            acc, acc_tgt = analyze_ae(X_test, X_adv, Y_test, preds, desc, y_target)

            save_images_and_estimates(X_adv, Y_test, preds, os.path.join(out_dir, 'Images', desc))
            #save_images_and_estimates(X_test - X_adv, Y_test, preds, os.path.join(out_dir, 'Images', desc))
            acc_ifgm[ord].append(acc)
            acc_tgt_ifgm[ord].append(acc_tgt)


    #--------------------------------------------------
    # Post-attack Analysis for *FGM
    #--------------------------------------------------
    for ord in [np.inf, 1, 2]:
        plt.plot(epsilon_map[ord], acc_fgm[ord], 'o-', label='FGM')
        plt.plot(epsilon_map[ord], acc_ifgm[ord], 'o-', label='I-FGM')
        plt.legend()
        plt.xlabel('epsilon')
        plt.ylabel('CNN accuracy')
        plt.title('ell_%s' % ord)
        plt.grid('on')
        plt.savefig('./output/attack_accuracy_%s.png' % ord, bbox_inches='tight')
        plt.close()
     
        plt.figure()
        plt.plot(epsilon_map[ord], acc_tgt_fgm[ord], 'o-', label='FGM')
        plt.plot(epsilon_map[ord], acc_tgt_ifgm[ord], 'o-', label='I-FGM')
        plt.legend()
        plt.xlabel('epsilon')
        plt.ylabel('Targeted AE Success Rate')
        plt.title('ell_%s' % ord)
        plt.grid('on')
        plt.savefig('./output/targeted_attack_accuracy_%s.png' % ord, bbox_inches='tight')
        plt.close()


    #--------------------------------------------------
    # Elastic Net
    # Note: this attack takes awhile to compute...(compared to *FGSM)
    #--------------------------------------------------
    attack = ElasticNetMethod(model_CH, sess=sess)
    c_vals = [1e-2, 1e-1, 1, 1e2, 1e4]
    acc_all_elastic = np.zeros((len(c_vals),))

    if 0:   # turn off for now, is slow
    #for idx, c in enumerate(c_vals):
        x_adv_tf = attack.generate(x_tf, 
                                   batch_size=batch_size,
                                   y_target=Y_target_OB, 
                                   beta=1e-3,            # ell_1 coeff
                                   confidence=1e-2,      # \kappa value from equation (4)
                                   initial_const=c,      # (an initial value for) c from eq. (7) - note this value increases as binary search progresses...
                                   clip_min=0.0,
                                   clip_max=c_max)

        #
        # Run the attack (targeted or untargeted)
        # on the test data.
        #
        if Y_target is not None:
            X_adv = run_in_batches(sess, x_tf, y_tf, x_adv_tf, X_test, Y_target, batch_size)
        else:
            X_adv = run_in_batches(sess, x_tf, y_tf, x_adv_tf, X_test, Y_test, batch_size)

        #
        # Evaluate the AE. 
        # Currently using the same model we originally attacked.
        #
        model_eval = model
        preds_tf = model_eval(x_tf)
        preds = run_in_batches(sess, x_tf, y_tf, preds_tf, X_adv, Y_test, batch_size)
        print('Test accuracy after E-Net attack: %0.2f' % calc_acc(Y_test, preds))
        print('Maximum per-pixel delta: %0.3f' % np.max(np.abs(X_test - X_adv)))
        print('Mean per-pixel delta: %0.3f' % np.mean(np.abs(X_test - X_adv)))
        print('l2: ', np.sqrt(np.sum((X_test - X_adv)**2)))
        print('l1: ', np.sum(np.abs(X_test - X_adv)))
        print(confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(preds, axis=1)))

        save_images_and_estimates(X_adv, Y_test, preds, os.path.join(out_dir, 'Images', 'Elastic_c%03d' % c))
        acc_all_elastic[idx] = calc_acc(Y_test, preds)


    #--------------------------------------------------
    # Saliency Map Attack
    # Note: this is *extremely* slow; will require overnight runs
    #--------------------------------------------------
    attack = SaliencyMapMethod(model_CH, sess=sess)
    acc_all_saliency = np.zeros((len(epsilon_values),))

    #for idx, epsilon in enumerate(epsilon_values):
    if False:
        x_adv_tf = attack.generate(x_tf, theta=epsilon/255., 
                                     y_target=y_tf,
                                     clip_min=0.0, 
                                     clip_max=255.0)

        #
        # Run the attack (targeted or untargeted)
        # on the test data.
        #
        if Y_target is not None:
            X_adv = run_in_batches(sess, x_tf, y_tf, x_adv_tf, X_test, Y_target, batch_size)
        else:
            X_adv = run_in_batches(sess, x_tf, y_tf, x_adv_tf, X_test, Y_test, batch_size)

        #
        # Evaluate the AE. 
        # Currently using the same model we originally attacked.
        #
        model_eval = model
        preds_tf = model_eval(x_tf)
        preds = run_in_batches(sess, x_tf, y_tf, preds_tf, X_adv, Y_test, batch_size)
        print('Test accuracy after SMM attack: %0.3f' % calc_acc(Y_test, preds))
        print('Maximum per-pixel delta: %0.1f' % np.max(np.abs(X_test - X_adv)))
        print(confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(preds, axis=1)))

        save_images_and_estimates(X_adv, Y_test, preds, os.path.join(out_dir, 'Images', 'Saliency_%02d' % epsilon))
        acc_all_saliency[idx] = calc_acc(Y_test, preds)


    #--------------------------------------------------
    # C&W ell-2
    #--------------------------------------------------
    if 0:
        attack = CarliniWagnerL2(model, sess=sess)
        x_adv_tf = attack.generate(x_tf, confidence=.1, y_target=Y_target_OB)




#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

def main(argv=None):
    # Set TF random seed to improve reproducibility
    np.random.seed(1066)
    tf.set_random_seed(1246)

    #--------------------------------------------------
    # Some experiment parameters
    #--------------------------------------------------
    gtsrb_image_dir = sys.argv[1] if len(sys.argv) > 1 else '~/Data/GTSRB/Final_Training/Images'
    target_class = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    output_dir = './output_gtsrb'
    cnn_weight_file = os.path.join(output_dir, 'gtsrb.ckpt')

    #--------------------------------------------------
    # load_data
    #--------------------------------------------------
    x_train, y_train, x_test, y_test = load_gtsrb_data(gtsrb_image_dir, output_dir)
    n_classes = np.max(y_train+1)

    y_train = to_categorical(y_train, n_classes)
    y_test = to_categorical(y_test, n_classes)

    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    #--------------------------------------------------
    # Train or attack CNN
    #--------------------------------------------------
    with tf.Session() as sess:
      if not os.path.exists(output_dir) or not tf.train.checkpoint_exists(cnn_weight_file):
          print("Training CNN")
          train_cnn(sess, (x_train, y_train, x_test, y_test), cnn_weight_file)
      else:
          print("Attacking CNN")
          attack_cnn(sess, (x_train, y_train, x_test, y_test), output_dir, cnn_weight_file, y_target=target_class)


if __name__ == '__main__':
    app.run()
