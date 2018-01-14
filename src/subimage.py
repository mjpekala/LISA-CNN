"""
Code to facilitate working with sub-images/regions within larger images.
"""

__author__ = "mjp"
__date__ = "dec 2017"


import os
import pdb

import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split



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

  
  

class Subimage(object):
  def __init__(self, to_grayscale=False):
    self._filenames = []
    self._bbox = []
    self._y = []
    self._gid = [] # group/track id - optional
    self._to_grayscale = to_grayscale


  def __str__(self):
    out = '%d images' % len(self._filenames)
    out += ' with %d unique classes' % len(np.unique(np.array(self._y)))
    return out


  def add(self, filename, bbox, y, gid=None):
    self._filenames.append(filename)
    self._bbox.append(bbox)
    self._y.append(y)
    self._gid.append(gid)


  def describe(self, indices):
    y = np.array(self._y)
    y = y[indices]
    out = '%d objects with %d unique class labels\n' % (len(indices), np.unique(y).size)
    for yi in np.unique(y):
      out += '  y=%d : %d\n' % (yi, np.sum(y == yi))
    return out


  def train_test_split(self, pct_test, max_per_class=np.Inf, reserve_for_test=[]):
    assert(pct_test < 1.0)

    # (optional): there may be certain images we want to explicitly reserve for test
    reserved_indices = []
    for pattern in reserve_for_test:
      for idx, filename in enumerate(self._filenames):
        if pattern in filename:
          reserved_indices.append(idx)
    reserved_indices = np.array(reserved_indices, dtype=np.int32)

    # generate the train/test split
    indices = np.delete(np.arange(len(self._y)), reserved_indices)
    class_labels = np.delete(np.array(self._y, dtype=np.int32), reserved_indices)

    train, test = train_test_split(indices, test_size=pct_test, stratify=class_labels)
    test = np.concatenate([test, np.array(reserved_indices, dtype=np.int32)])

    # (optional): limit max # of examples in a given class (for training)
    if np.isfinite(max_per_class):
      y = np.array(self._y)

      for yi in np.unique(y):
        yi_idx = np.nonzero(y[train] == yi)[0]
        if len(yi_idx) > max_per_class:
          train = np.delete(train, yi_idx[max_per_class:])

    return train, test


  def get_images(self, indices):
    "Returns full sized images at the specified indices."
    out = []
    for idx in indices:
      im = Image.open(self._filenames[idx])

      if self._to_grayscale:
        im = im.convert('L')

      out.append(np.array(im))

    y = np.array(self._y)
    return out, y[indices]


  def get_subimages(self, indices, new_size=None, pct_context=0, verbose=False):
    "Extracts sub-indices from images."
    
    out = []
    for ii, idx in enumerate(indices):
      if verbose and np.mod(ii, 500) == 0:
        print('loading image %d (of %d)' % (ii, len(indices)))
        
      im = Image.open(self._filenames[idx])
      if self._to_grayscale:
        im = im.convert('L')

      bbox = self._bbox[idx]

      # (optional) expand box to grab additional context.
      # currently assumes sub-images are well within the interior of the image.
      if pct_context > 0:
        dx = bbox[2] - bbox[0]
        dy = bbox[3] - bbox[1]
        dx_new = dx * (1. + pct_context)
        dy_new = dy * (1. + pct_context)

        x0 = np.floor((bbox[2] + bbox[0])/2 - dx_new/2.)
        x1 = np.floor((bbox[2] + bbox[0])/2 + dx_new/2.)
        y0 = np.floor((bbox[3] + bbox[1])/2 - dy_new/2.)
        y1 = np.floor((bbox[3] + bbox[1])/2 + dy_new/2.)

        bbox = [int(x0), int(y0), int(x1), int(y1)]

      # crop out the sub-image
      im = im.crop(bbox)

      # (optional) resize subimage
      if new_size is not None:
        im = im.resize(new_size, Image.ANTIALIAS)

      im_arr = np.array(im)
      if im_arr.ndim == 2:
        im_arr = im_arr[:,:,np.newaxis] # force channel dimension

      out.append(np.array(im_arr))

    y = np.array(self._y)
    return out, y[indices]


  def splice_subimages(self, indices, new_subimage):
    """
       Splice new subimages into original images.

         new_subimage : either (a) a list of images to inject or 
                               (b) a tensor of images (n x rows x cols x channels)
    """
    out = []

    for ii, idx in enumerate(indices):
      # extract the ith sub-image
      if isinstance(new_subimage, list) or isinstance(new_subimage, tuple):
        si = new_subimage[ii]
      else:
        si = new_subimage[ii,...]
      assert(si.ndim >= 2)

      # corresponding full scene
      im = Image.open(self._filenames[idx])
      if self._to_grayscale:
        im = im.convert('L')
      xi = np.array(im)

      # the bounding box
      x0,y0,x1,y1 = self._bbox[idx]
      width = x1-x0
      height = y1-y0

      # resize new image (if needed) and blast into bounding box
      if si.shape[0] != height or si.shape[1] != width:
        si = Image.fromarray(si).resize((width,height), Image.ANTIALIAS)
        si = np.array(si)

      xi[y0:y1,x0:x1] = np.squeeze(si)
      out.append(xi)

    return out

#-------------------------------------------------------------------------------


def _test_splicing():
  si = parse_LISA('~/Data/LISA/allAnnotations.csv')

  # ensures that, without resizing, one can paste the
  # exact sub-image back into the original image
  indices = np.arange(100)
  x_big, _ = si.get_images(indices)
  x_small, _ = si.get_subimages(indices, None)
  x_splice = si.splice_subimages(indices, x_small)

  for ii in range(len(x_splice)):
    assert(np.all(x_splice[ii] == x_big[ii]))

  return si


def _test_reserve_images():
  si = parse_LISA('~/Data/LISA/allAnnotations.csv')
  prefix = 'stop_1323803184.avi_image'
  train, test = si.train_test_split(.17, max_per_class=500, reserve_for_test=[prefix,])

  for idx in train:
    assert(not prefix in si._filenames[idx])



if __name__ == "__main__":
  # we need some data for testing; just use LISA for now.
  from lisa import parse_LISA
  
  _test_splicing() 
  _test_reserve_images()

  si = parse_LISA('~/Data/LISA/allAnnotations.csv')

  # this should approximate table I in Evtimov et al. fairly closely
  train_idx, test_idx = si.train_test_split(.17, max_per_class=500)
  print(si.describe(train_idx))
  print(si.describe(test_idx))

  # save sub-images to file for manual inspection
  print('extracting sub-images...')
  x_test, y_test = si.get_subimages(test_idx, (32,32), pct_context=.5)
  x_test = np.array(x_test) # [] -> tensor
  np.savez('test_images.npz', x_test=x_test, y_test=y_test)
