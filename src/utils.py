"""  Helper functions.
"""

import os

import numpy as np

import tensorflow as tf
import keras


#-------------------------------------------------------------------------------
# Generic helper/utility functions
#-------------------------------------------------------------------------------

def splitpath(full_path):
  """
  Splits a path into all possible pieces (vs. just head/tail).
  """
  head, tail = os.path.split(full_path)

  result = [tail]

  while len(head) > 0:
    [head, tail] = os.path.split(head)
    result.append(tail)

  result = [x for x in result if len(x)]
  return result[::-1]



def makedirs_if_needed(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

        
#-------------------------------------------------------------------------------
# Functions related to training/testing CNNs
#-------------------------------------------------------------------------------


def to_categorical(y, num_classes, dtype=np.float32, smooth=False):
    """ Converts a vector of integer class labels into a one-hot matrix representation.

      e.g.  [1 3 7] becomes:

          [0 1 0 0 0 0 0 0 0 0 0 ;
           0 0 0 1 0 0 0 0 0 0 0 ;
           0 0 0 0 0 0 0 1 0 0 0 ]
    """
    out = np.zeros((y.size, num_classes), dtype=dtype)

    for ii in range(y.size):
        out[ii,y[ii]] = 1

    if smooth:
      # put .95 weight on true class, divide rest among other classes
      nz_value = 0.05 / (num_classes-1)
      out[out==0] = nz_value
      out[out==1] = 0.95

    return out



def categorical_matrix(y_scalar, num_copies, num_classes, *args, **kargs):
    """ Creates a matrix of one-hot target class labels
        (for use with targeted attacks).  

        e.g. categorical_matrix(1,3,10) is

          [ 0 1 0 0 0 0 0 0 0 0 ;
            0 1 0 0 0 0 0 0 0 0 ;
            0 1 0 0 0 0 0 0 0 0 ]
    """
    y_tgt = y_scalar * np.ones((num_copies,), dtype=np.int32)
    return to_categorical(y_tgt, num_classes, *args, **kargs)



def calc_acc(y_true_OH, y_hat_OH):
    """ Computes classification accuracy from a pair of one-hot estimates/truth.

    It doesn't really matter if the arguments are one-hot; just need the
    argmax along dimension 1 gives the estimated class label.

    """
    is_correct = np.argmax(y_hat_OH, axis=1) == np.argmax(y_true_OH, axis=1)
    return 100. * np.sum(is_correct) / y_hat_OH.shape[0]



def analyze_ae(X_orig, X_adv, Y_true_OH, preds_OH, desc, y_tgt=None):
    """ Displays information re. the effectiveness of AE to stdout.
    """
    acc = calc_acc(Y_true_OH, preds_OH)
    acc_tgt = None

    print('Results for %s:' % desc)
    print('    CNN Accuracy: %0.2f' % calc_acc(Y_true_OH, preds_OH))

    if y_tgt is not None:
        y_syn = categorical_matrix(y_tgt, Y_true_OH.shape[0], Y_true_OH.shape[1])
        acc_tgt = calc_acc(y_syn, preds_OH)
        print('    targeted AE success rate: %0.2f' % acc_tgt)

    print('    ||x-x_ae||_inf: %0.3f' % np.max(np.abs(X_orig - X_adv)))
    print('    ||x-x_ae||_1  : %0.3f' % np.sum(np.abs(X_orig - X_adv)))
    print('    ||x-x_ae||_2  : %0.3f' % np.sum(np.sqrt((X_orig - X_adv)**2)))

    print(confusion_matrix(np.argmax(Y_true_OH, axis=1), np.argmax(preds_OH, axis=1)))

    return acc, acc_tgt



def run_in_batches(sess, x_tf, y_tf, output_tf, x_in, y_in, batch_size):
    """ 
     Runs data through a CNN one batch at a time; gathers all results
     together into a single tensor.  This assumes the output of each
     batch is tensor-like.

        sess      : the tensorflow session to use
        x_tf      : placeholder for input x
        y_tf      : placeholder for input y
        output_tf : placeholder for CNN output
        x_in      : data set to process (numpy tensor)
        y_in      : associated labels (numpy, one-hot encoding)
        batch_size : minibatch size (scalar)

    """
    n_examples = x_in.shape[0]  # total num. of objects to feed

    # determine how many mini-batches are required
    nb_batches = int(math.ceil(float(n_examples) / batch_size))
    assert nb_batches * batch_size >= n_examples

    out = []
    with sess.as_default():
        for start in np.arange(0, n_examples, batch_size):
            # the min() stuff here is to handle the last batch, which may be partial
            end = min(n_examples, start + batch_size)
            start_actual = min(start, n_examples - batch_size)

            feed_dict = {x_tf : x_in[start_actual:end], y_tf : y_in[start_actual:end]}
            output_i = sess.run(output_tf, feed_dict=feed_dict)

            # the slice is to avoid any extra stuff in last mini-batch,
            # which might not be entirely "full"
            skip = start - start_actual
            output_i = output_i[skip:]
            out.append(output_i)

    out = np.concatenate(out, axis=0)
    assert(out.shape[0] == n_examples)
    return out



def save_images_and_estimates(x, y_true_OH, y_est_OH, base_dir, y_to_classname=None):
    correct_dir = os.path.join(base_dir, 'correctly_classified')
    incorrect_dir = os.path.join(base_dir, 'incorrectly_classified')

    if np.max(x) <= 1.0:
        x = x * 255.

    for idx in range(x.shape[0]):
        y_true = np.argmax(y_true_OH[idx,...])
        y_est = np.argmax(y_est_OH[idx,...])

        y_true_str = y_to_classname[y_true] if y_to_classname is not None else str(y_true)
        y_est_str = y_to_classname[y_est] if y_to_classname is not None else str(y_est)

        fn = 'image_%05d_y%02d_%s.png' % (idx, y_true, y_true_str)

        if y_true == y_est:
            out_dir = os.path.join(correct_dir, y_est_str)
        else:
            out_dir = os.path.join(incorrect_dir, y_est_str)
        makedirs_if_needed(out_dir)

        img = Image.fromarray(np.squeeze(x[idx]).astype('uint8'))
        img.save(os.path.join(out_dir, fn))


        
def simple_cnn_model(input_shape=(32,32,1), nb_filters=64, nb_classes=48):
    """
    A slight variation on cleverhans.utils_keras.cnn_model()
    """
    #model = Sequential()

    def normalize(x):
        mu = tf.reduce_mean(x, axis=2, keep_dims=True)
        return x - mu
        #_, sigma = tf.nn.moments(x, axes=[0])  # along mini-batch dimension
        #return (x - mu) / sigma

        
    # Define the layers successively (convolution layers are version dependent)
    assert(keras.backend.image_dim_ordering() == 'tf')

    image_input = keras.layers.Input(shape=input_shape)

    # Note: I assume here the backend is tensorflow!
    #x = keras.layers.Lambda(lambda x: x - tf.reduce_mean(x, axis=2, keep_dims=True))(image_input)
    x = keras.layers.Lambda(normalize)(image_input)

    x = keras.layers.Conv2D(nb_filters, kernel_size=(8,8), strides=(2,2), padding="same")(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(nb_filters, kernel_size=(6,6), strides=(2,2), padding="valid")(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(nb_filters*2, kernel_size=(5,5), strides=(1,1), padding="valid")(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(nb_classes)(x)
    predictions = keras.layers.Activation('softmax')(x)

    model = keras.models.Model(inputs=image_input, outputs=predictions)

    return model

