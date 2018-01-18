# Adversarial Examples (AE) for Street Sign Data

This module provides some engineering (i.e. quick-and-dirty) code for generating AE in the context of street sign classification problems.  The current experiment is using the LISA street sign data set and the Cleverhans (CH) library for adversarial attacks.

## Quick Start

The [Makefile](./Makefile) provides examples of how to do everything.  Note that there are a few assumptions about the local configuration baked in (e.g. you have keras, tensorflow, LISA dataset in ~/Data/LISA, etc.).  To run the experiments, it should only be necessary to run "make" a few times (or do the equivalent operations manually).  The first invocation of "make" will checkout a copy of Cleverhans library (if needed), extract street sign images from the LISA dataset and train a simple, 3 layer CNN.  The second invocation of "make" will then generate AE for this pre-trained model.  Results (e.g. model weights, AE images, etc.) will all be stored in "./output".  

If you have your own tensorflow model and just want to attack that, take a look in [lisacnn.py](./lisacnn.py) at the function attack_lisa_cnn().  This provides one example of how to use the library.  Alternately, there are a few examples in the Cleverhans codes itself.

If you have any questions, don't hesitate to ask!
