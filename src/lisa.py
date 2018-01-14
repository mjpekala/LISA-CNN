"""
Code for working with LISA dataset.
"""

import os
import numpy as np

from subimage import Subimage


#-------------------------------------------------------------------------------
# These correspond to table 1 from Evtimov et al. "Robust Physical-World Attacks on Deep Learning Models".
# It does *not* capture all possible classes in LISA.

LISA_17_CLASSES = ["addedLane", "keepRight", "laneEnds", "merge", "pedestrianCrossing", "school",
                   "schoolSpeedLimit25", "signalAhead", "speedLimit25", "speedLimit30", "speedLimit35",
                   "speedLimit45", "speedLimitUrdbl", "stop", "stopAhead", "turnRight", "yield"]

LISA_17_CLASS_MAP = { x : ii for ii,x in enumerate(LISA_17_CLASSES) }


# This is the subset of "conventional" speed limit signs
LISA_SPEED_LIMIT_CLASSES = ['speedLimit25', 'speedLimit30', 'speedLimit35', 'speedLimit40',
                            'speedLimit45', 'speedLimit50', 'speedLimit55', 'speedLimit60', 'speedLimit65']

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



def default_train_test_split(si):
    """ Generates a track-based train/test split.

        si : a Subimage object corresponding to the LISA data set.
    """
    train_grp = ['aiua120214-0', 'aiua120214-1', 'aiua120306-1'] + ['vid%d' % x for x in range(6)]
    test_grp = ['aiua120214-2', 'aiua120306-0'] + ['vid%d' % x for x in range(6,12)]

    train_idx = []
    test_idx = []

    for gid in train_grp:
        indices = [x for x in range(len(si._y)) if si._gid[x] == gid]
        assert(len(indices) > 0)
        train_idx.extend(indices)
    
    for gid in test_grp:
        indices = [x for x in range(len(si._y)) if si._gid[x] == gid]
        assert(len(indices) > 0)
        test_idx.extend(indices)

    train_idx = np.array(train_idx, np.int32)
    test_idx = np.array(test_idx, np.int32)

    # make sure (train, test) partition the total set of images
    assert(np.setdiff1d(train_idx, test_idx).size == train_idx.size)
    assert(test_idx.size + train_idx.size == len(si._y))

    return train_idx, test_idx

    

def parse_annotations(csvfile, class_map=LISA_17_CLASS_MAP):
    """ See also: tools/extractAnnotations.py in LISA dataset.
    """
  
    si = Subimage(to_grayscale=True)

    # annotations are stored in a master CSV (well, semicolon-delimited) file
    csvfile = os.path.expanduser(csvfile)
    csv = open(csvfile, 'r')
    csv.readline() # discard header
    csv = csv.readlines()

    # Note: the original LISA parsing code shuffled the rows, but this just adds 
    #       potential confusion so I'm not doing that for now.

    # If no classmap was provided, create one.
    if class_map is None:
        class_map = {}
        for line in csv:
            fields = line.split(';')
            
            if fields[1] not in class_map:
                class_map[fields[1]] = len(class_map)
  
    # base path to actual filenames.
    base_path = os.path.dirname(csvfile)

    # do it
    for idx, line in enumerate(csv):
        fields = line.split(';')
        y_str = fields[1]
        if y_str not in class_map:
            continue

        im_filename = fields[0]
        gid = splitpath(im_filename)[0]  # update: use top level directory as group id
        y = class_map[y_str]
        x0 = int(fields[2])
        x1 = int(fields[4])
        y0 = int(fields[3])
        y1 = int(fields[5])
        bbox = [x0,y0,x1,y1]

        si.add(os.path.join(base_path, im_filename), bbox, y, gid)

    return si

