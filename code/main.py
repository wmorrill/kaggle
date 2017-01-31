__author__ = 'wmorrill'
# Thought Process:
# take training set and isolate lung with a smooth surface (we want to catch nodules that might be on the edge)
# probably add some buffer to this volume
# chop the lung volume into cubes (or possibly just search the whole image?)
# check each cube for a sphere like object
# use training set to determine which sphere like objects are not cancer and which might be
# (we only know which patients have chance not where it is or what it looks like)
# determine the difference in features between definite negatives and potential positives
# possible features; size, uniformity, density,


# Location of info to get tensorflow working
# http://www.heatonresearch.com/2017/01/01/tensorflow-windows-gpu.html

# retrain inception v3 on 2d images?

