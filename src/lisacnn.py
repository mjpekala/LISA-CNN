#!/bin/env python

""" Code for trainign and then attacking a simple street sign detector based on the LISA data set.

 References:
   o LISA data set http://cvrr.ucsd.edu/LISA/lisa-traffic-sign-dataset.html
"""


import os
import random
import math
import json
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
import lisa


CLASSES = lisa.LISA_17_CLASSES
CLASS_MAP = lisa.LISA_17_CLASS_MAP


FLAGS = flags.FLAGS

flags.DEFINE_string('train_dir', 'output', 'Directory storing the saved model and other outputs (e.g. adversarial images).')
flags.DEFINE_string('data_dir','/home/neilf/Fendley/data/signDatabase/annotations', 'The Directory in which the extra lisadataset is')
flags.DEFINE_string(
    'filename', 'lisacnn.ckpt', 'Filename to save model under.')
flags.DEFINE_integer('nb_classes', 48, 'Number of classes')
flags.DEFINE_integer('nb_epochs', 80, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training') # use this for values in [0,1]
flags.DEFINE_float('epsilon', 20, 'FGSM perturbation constraint')
flags.DEFINE_bool('DO_CONF', False, 'Generate the confusion matrix on the test set')
flags.DEFINE_bool('DO_ADV', True, 'Generate the adversarial examples on the test set')
flags.DEFINE_bool('force_retrain', False, 'Ignore if you have already trained a model')
flags.DEFINE_bool('save_adv_img', True, 'Save the adversarial images generated on the test set')




#-------------------------------------------------------------------------------
# LISA-CNN codes
#-------------------------------------------------------------------------------


def load_lisa_data(with_context=True):
    """
    Loads LISA data set.
    
    FLAGS:
        annotation_file: main LISA annotation file

    RETURNS:
        xtrain: numpy array of training data
        ytrain: numpy array of traning labels
        xtest: numpy array of test data
        ytest: numpy array of test labels
    """
    annotation_file = '~/Data/LISA/allAnnotations.csv'
    cache_fn = './output/lisa_data.npz'

    # Create the dataset if it does not already exist.
    # Note that we create it once, up front, so that we
    # can ensure a consistent train/test split across all experiments.
    if not os.path.exists(cache_fn):
        print('[load_lisa_data]: Extracting sign images from video frames...please wait...')

        si = lisa.parse_annotations(annotation_file, CLASS_MAP)
        #train_idx, test_idx = si.train_test_split(.17, max_per_class=500, reserve_for_test=['stop_1323803184.avi_image'])
        train_idx, test_idx = lisa.default_train_test_split(si, max_per_class=500)
        print(si.describe(train_idx))
        print(si.describe(test_idx))

        #
        # signs *without* extra context
        #
        x_train, y_train = si.get_subimages(train_idx, (32,32), 0.0)
        x_train = np.array(x_train) # list -> tensor
        y_train = np.array(y_train)

        x_test, y_test = si.get_subimages(test_idx, (32,32), 0.0)
        x_test = np.array(x_test) # list -> tensor
        y_test = np.array(y_test)

        #
        # signs *with* extra context
        #
        #sz = 38
        sz = 42
        pct = float(sz-32)/sz
        x_train_c, y_train_c = si.get_subimages(train_idx, (sz,sz), pct)
        x_train_c = np.array(x_train_c)
        y_train_c = np.array(y_train_c)

        x_test_c, y_test_c = si.get_subimages(test_idx, (sz,sz), pct)
        x_test_c = np.array(x_test_c)
        y_test_c = np.array(y_test_c) 

        # (optional) rescale
        # with the default hyperparameters, this actually hurts performance...
        if 1:
            x_train = x_train.astype(np.float32) / 255.
            x_test = x_test.astype(np.float32) / 255.
            x_train_c = x_train_c.astype(np.float32) / 255.
            x_test_c = x_test_c.astype(np.float32) / 255.

        # save for quicker reload next time
        makedirs_if_needed(os.path.dirname(cache_fn))
        np.savez(cache_fn, train_idx=train_idx,  
                           test_idx=test_idx, 
                           x_train=x_train, 
                           y_train=y_train, 
                           x_test=x_test, 
                           y_test=y_test,
                           x_train_context=x_train_c, 
                           y_train_context=y_train_c,
                           x_test_context=x_test_c, 
                           y_test_context=y_test_c)

    f = np.load(cache_fn)

    if with_context:
        x_train = f['x_train_context']
        y_train = f['y_train_context']
        x_test = f['x_test_context']
        y_test = f['y_test_context']
    else:
        x_train = f['x_train']
        y_train = f['y_train']
        x_test = f['x_test']
        y_test = f['y_test']


    for string, data in zip(['train data', 'train labels', 'test data', 'test labels'], [x_train, y_train, x_test, y_test]):
        print('[load_lisa_data]: ', string, data.shape, data.dtype, np.min(data), np.max(data))

    return x_train, y_train, x_test, y_test



def data_lisa(with_context):
    """
    Funtion to read in the data prepared by the lisa dataset
    The train test split will be randomly generated, or loaded if you have a /tmp 
    split saved already
    Returns:
        xtrain: numpy array of the training data 
        ytrain: categorical array of training labels
        xtest: numpy array of the test data
        ytest: numpy array of the test labels
    
    """
    # NOTE: it would appear the LISA annotation extraction code introduces some 
    #       label noise.  Therefore, we do the extraction ourselves.
    X_train, Y_train, X_test, Y_test = load_lisa_data(with_context)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Note: we moved the affine rescaling directly into the network

    # Downsample to promote class balance during training
    #X_train, Y_train = downsample_data_set(X_train, Y_train, per_class_limit)

    print('X_train shape:', X_train.shape)
    print('Y_train shape:', Y_train.shape)
    print('X_test shape: ', X_test.shape)

    all_labels = np.unique(Y_train)
    print('There are %d unique classes:' % all_labels.size)
    for yi in all_labels:
        print('  y=%d : %d' % (yi, np.sum(Y_train==yi)))

    Y_train = to_categorical(Y_train, FLAGS.nb_classes)
    Y_test = to_categorical(Y_test, FLAGS.nb_classes)

    return X_train, Y_train, X_test, Y_test



def make_lisa_cnn(sess, batch_size, dim):
    """  Creates a simple sign classification network.
    
    Note that the network produced by cnn_model() is fairly weak.
    For example, on CIFAR-10 it gets 60-something percent accuracy,
    which substantially below state-of-the-art.

    Note: it is not required here that dim be the same as the
    CNN input spatial dimensions.  In these cases, the
    caller is responsible for resizing x to make it
    compatible with the model (e.g. via random crops).
    """
    num_classes=48  # usually only use 17 vs 48 classes, but it doesn't hurt to have room for 48
    num_channels=1
    x = tf.placeholder(tf.float32, shape=(batch_size, dim, dim, num_channels))
    y = tf.placeholder(tf.float32, shape=(batch_size, num_classes))

    # XXX: set layer naming convention explicitly? 
    #      Otherwise, names depend upon when model was created...
    #model = cnn_model(img_rows=32, img_cols=32, channels=num_channels, nb_classes=num_classes)
    model = simple_cnn_model()

    return model, x, y



def train_lisa_cnn(sess, cnn_weight_file):
    """ Trains the LISA-CNN network.
    """
    X_train, Y_train, X_test, Y_test = data_lisa(with_context=True)

    model, x, y = make_lisa_cnn(sess, FLAGS.batch_size, X_train.shape[1])

    # construct an explicit predictions variable
    x_crop = tf.random_crop(x, (FLAGS.batch_size, 32, 32, X_train.shape[-1]))
    model_output = model(x_crop)

    def evaluate():
        # Evaluate accuracy of the lisaCNN model on clean test examples.
        preds = run_in_batches(sess, x, y, model_output, X_test, Y_test, FLAGS.batch_size)
        print('test accuracy: ', calc_acc(Y_test, preds))

    # Note: model_train() will add some new (Adam-related) variables to the graph
    train_params = {
        'nb_epochs': FLAGS.nb_epochs,
        'batch_size': FLAGS.batch_size,
        'learning_rate': FLAGS.learning_rate
    }
    model_train(sess, x, y, model_output, X_train, Y_train, evaluate=evaluate, args=train_params)

    saver = tf.train.Saver()
    save_path = saver.save(sess, cnn_weight_file)
    print("Model was saved to " +  cnn_weight_file)




def attack_lisa_cnn(sess, cnn_weight_file, y_target=None, standardize=True):
    """ Generates AE for the LISA-CNN.
        Assumes you have already run train_lisa_cnn() to train the network.
    """
    epsilon_map = {np.inf : [.02, .05, .075, .1, .15, .2],   # assumes values in [0,1]
                        1 :      [.1, 1, 10], 
                        2 :      [.1, 1, 10]}

    #--------------------------------------------------
    # data set prep
    #--------------------------------------------------
    # Note: we load the version of the data *without* extra context
    X_train, Y_train, X_test, Y_test = data_lisa(with_context=False)

    # Create one-hot target labels (needed for targeted attacks only)
    if y_target is not None:
        Y_target_OB = categorical_matrix(y_target, FLAGS.batch_size, Y_test.shape[1])
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
    model, x_tf, y_tf = make_lisa_cnn(sess, FLAGS.batch_size, X_train.shape[1])
    model_CH = KerasModelWrapper(model) # to make CH happy

    # the input may or may not require some additional transformation
    if standardize:
        x_input = tf.map_fn(lambda z: per_image_standardization(z), x_tf)
    else:
        x_input = x_tf
    model_output = model(x_input)


    saver = tf.train.Saver()
    saver.restore(sess, cnn_weight_file)

    #--------------------------------------------------
    # Performance on clean data
    # (try this before attacking)
    #--------------------------------------------------
    predictions = run_in_batches(sess, x_tf, y_tf, model_output, X_test, Y_test, FLAGS.batch_size)
    acc_clean = calc_acc(Y_test, predictions)
    print('[info]: accuracy on clean test data: %0.2f' % acc_clean)
    print(confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(predictions, axis=1)))

    save_images_and_estimates(X_test, Y_test, predictions, 'output/Images/Original', CLASSES)


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
                X_adv = run_in_batches(sess, x_tf, y_tf, x_adv_tf, X_test, Y_target, FLAGS.batch_size)
            else:
                X_adv = run_in_batches(sess, x_tf, y_tf, x_adv_tf, X_test, Y_test, FLAGS.batch_size)

            #
            # Evaluate the AE. 
            # Currently using the same model we originally attacked.
            #
            model_eval = model
            #preds_tf = model_eval(x_tf)
            preds_tf = model_eval(x_input)
            preds = run_in_batches(sess, x_tf, y_tf, preds_tf, X_adv, Y_test, FLAGS.batch_size)
            acc, acc_tgt = analyze_ae(X_test, X_adv, Y_test, preds, desc, y_target)

            save_images_and_estimates(X_adv, Y_test, preds, 'output/Images/%s' % desc, CLASSES)
            save_images_and_estimates(X_test - X_adv, Y_test, preds, 'output/Deltas/%s' % desc, CLASSES)
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
                X_adv = run_in_batches(sess, x_tf, y_tf, x_adv_tf, X_test, Y_target, FLAGS.batch_size)
            else:
                X_adv = run_in_batches(sess, x_tf, y_tf, x_adv_tf, X_test, Y_test, FLAGS.batch_size)

            #
            # Evaluate the AE. 
            # Currently using the same model we originally attacked.
            #
            model_eval = model
            #preds_tf = model_eval(x_tf)
            preds_tf = model_eval(x_input)
            preds = run_in_batches(sess, x_tf, y_tf, preds_tf, X_adv, Y_test, FLAGS.batch_size)
            acc, acc_tgt = analyze_ae(X_test, X_adv, Y_test, preds, desc, y_target)

            save_images_and_estimates(X_adv, Y_test, preds, 'output/Images/%s' % desc, CLASSES)
            save_images_and_estimates(X_test - X_adv, Y_test, preds, 'output/Deltas/%s' % desc, CLASSES)
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
                                   batch_size=FLAGS.batch_size,
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
            X_adv = run_in_batches(sess, x_tf, y_tf, x_adv_tf, X_test, Y_target, FLAGS.batch_size)
        else:
            X_adv = run_in_batches(sess, x_tf, y_tf, x_adv_tf, X_test, Y_test, FLAGS.batch_size)

        #
        # Evaluate the AE. 
        # Currently using the same model we originally attacked.
        #
        model_eval = model
        preds_tf = model_eval(x_tf)
        preds = run_in_batches(sess, x_tf, y_tf, preds_tf, X_adv, Y_test, FLAGS.batch_size)
        print('Test accuracy after E-Net attack: %0.2f' % calc_acc(Y_test, preds))
        print('Maximum per-pixel delta: %0.3f' % np.max(np.abs(X_test - X_adv)))
        print('Mean per-pixel delta: %0.3f' % np.mean(np.abs(X_test - X_adv)))
        print('l2: ', np.sqrt(np.sum((X_test - X_adv)**2)))
        print('l1: ', np.sum(np.abs(X_test - X_adv)))
        print(confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(preds, axis=1)))

        save_images_and_estimates(X_adv, Y_test, preds, 'output/Images/Elastic_c%03d' % c, CLASSES)
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
            X_adv = run_in_batches(sess, x_tf, y_tf, x_adv_tf, X_test, Y_target, FLAGS.batch_size)
        else:
            X_adv = run_in_batches(sess, x_tf, y_tf, x_adv_tf, X_test, Y_test, FLAGS.batch_size)

        #
        # Evaluate the AE. 
        # Currently using the same model we originally attacked.
        #
        model_eval = model
        preds_tf = model_eval(x_tf)
        preds = run_in_batches(sess, x_tf, y_tf, preds_tf, X_adv, Y_test, FLAGS.batch_size)
        print('Test accuracy after SMM attack: %0.3f' % calc_acc(Y_test, preds))
        print('Maximum per-pixel delta: %0.1f' % np.max(np.abs(X_test - X_adv)))
        print(confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(preds, axis=1)))

        save_images_and_estimates(X_adv, Y_test, preds, 'output/Images/Saliency_%02d' % epsilon, CLASSES)
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

    # the CNN weight file
    cnn_weight_file = os.path.join(FLAGS.train_dir, FLAGS.filename)


    with tf.Session() as sess:
      if not os.path.exists(FLAGS.train_dir) or not tf.train.checkpoint_exists(cnn_weight_file) or FLAGS.force_retrain:
          print("Training LISA-CNN")
          train_lisa_cnn(sess, cnn_weight_file)
      else:
          print("Attacking LISA-CNN")
          attack_lisa_cnn(sess, cnn_weight_file, y_target=CLASS_MAP['speedLimit45'])
          #attack_lisa_cnn(sess, cnn_weight_file, y_target=None)


if __name__ == '__main__':
    app.run()
