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



def at_most_n_per_class(y, n):
    y_all = np.unique(y)
    indices_to_keep = []

    for yi in y_all:
        indices = np.nonzero(y == yi)[0]

        if indices.size <= n:
            indices_to_keep.append(indices)
        else:
            subset = np.random.choice(indices, n)
            indices_to_keep.append(subset)

    indices_to_keep = np.concatenate(indices_to_keep)
    return indices_to_keep

 


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

  
  def __len__(self):
    return len(self._y)

  
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


  def group_info(self):
    """ Provides some information related to group structure.
    """
    group_to_class = {}
    group_to_count = {}
    
    for group_id, y in zip(self._gid, self._y):
      # increment # of examples for this track by one
      group_to_count[group_id] = 1 if group_id not in group_to_count else group_to_count[group_id] + 1

      # store/check the class label for this track
      if group_id not in group_to_class:
        group_to_class[group_id] = y
      elif group_to_class[group_id] != y:
        print('ERROR: ', group_id, y, group_to_class[group_id])

    return group_to_class, group_to_count
          


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

 

  def train_test_split_by_group(self, pct_test, max_per_class=None):
    """ Provides a train/test split whereby a given group g is contained entirely within
        exactly one of train or test.  

        Note that the percentage test in this context is in terms of groups; if groups have
        vastly different cardinality the per-example distribution may be very different.

        This method returns train/test indices on a per-example basis (where group membership
        is respected as per the above).
    """
    assert(pct_test < 1.0)

    # XXX: could try to use group cardinality to balance per-example (future feature)
    grp_to_y, _ = self.group_info()

    all_groups = list(grp_to_y.keys())
    y_for_groups = [grp_to_y[g] for g in all_groups]

    # generate the train/test split
    train_groups_idx, test_groups_idx = train_test_split(range(len(all_groups)), test_size=pct_test, stratify=y_for_groups)


    # map group indices back to individual example indices
    train_idx = []
    for g_idx in train_groups_idx:
      indices = [i for i,x in enumerate(self._gid) if x == all_groups[g_idx]]
      train_idx.extend(indices)

    test_idx = []
    for g_idx in test_groups_idx:
      indices = [i for i,x in enumerate(self._gid) if x == all_groups[g_idx]]
      test_idx.extend(indices)


    # downsample so that we have no more than N per class (optional)
    if max_per_class is not None:
      y_train = [self._y[x] for x in train_idx]
      train_idx = [train_idx[x] for x in at_most_n_per_class(np.array(y_train), max_per_class)]

      y_test = [self._y[x] for x in test_idx]
      test_idx = [test_idx[x] for x in at_most_n_per_class(np.array(y_test), max_per_class)]

      
    # make certain these two sets are disjoint (on a per-example basis)
    assert(np.intersect1d(np.array(train_idx), np.array(test_idx)).size == 0)
    
    return train_idx, test_idx


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
