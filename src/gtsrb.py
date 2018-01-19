"""
  Code for working with GTSRB dataset.

  Note that these are RGB images; however, we convert to grayscale for compatability
  with an anticipated target platform.

  References:
     http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset
"""

import os
import numpy as np

from subimage import Subimage



def default_train_test_split(si):
    """ GTSRB has a train/test set; however, the test data comes without labels.
        This function can be used to split up the training set (e.g. into train and validation).

        si : a Subimage object 
    """
    np.random.seed(1066)
    return si.train_test_split_by_group(0.2, max_per_class=None)


def parse_train(train_dir):
    """ The training data is organized as one subdirectory per class.
    """
    train_dir = os.path.expanduser(train_dir)
    all_subdirs = [f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir,f))]

    # process each subdirectory
    si = Subimage(to_grayscale=True)
    for subdir in all_subdirs:
        csvfile = os.path.join(train_dir, subdir, 'GT-%s.csv' % subdir)
        _parse_annotations(csvfile, si)
        
    return si



def parse_test(test_csv):
    """  The test data is not organized into subirectories like the training.  
         Also, the filenames are different.  
    """
    si = Subimage(to_grayscale=True)
    _parse_annotations(test_csv, si)
    return si



def _parse_annotations(csvfile, si):
    """   Adds data from a directory to a subimage object
    """
  
    # annotations are stored in a master CSV (well, semicolon-delimited) file
    csvfile = os.path.expanduser(csvfile)
    csv = open(csvfile, 'r')
    csv.readline() # discard header
    csv = csv.readlines()

    # base path to actual filenames.
    base_path = os.path.dirname(csvfile)

    # do it
    for idx, line in enumerate(csv):
        fields = line.split(';')

        im_filename = fields[0]
        width, height = int(fields[1]), int(fields[2])
        x0, y0 = int(fields[3]), int(fields[4])
        x1, y1 = int(fields[5]), int(fields[6])
        bbox = [x0,y0,x1,y1]

        # test data has no labels
        if len(fields) > 7:
            y = int(fields[7])
        else:
            y = -1

        # Training data filenames have track and frame information.
        # Test data does not.
        if '_' in im_filename:
            # toss the .ppm and pull out track and frame id
            track_id, frame_id = im_filename[:-5].split('_')
            
            # The group id is a combination of the track id and the class label.
            gid = track_id + '_' + str(y)
        else:
            # There is no notion of a "track" in the test data
            gid = None

        si.add(os.path.join(base_path, im_filename), bbox, y, gid)

    return si

