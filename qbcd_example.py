# -*- coding: utf-8 -*-
# Create by Kaijun WANG in December 2023

import numpy as np
from qbcd_bias_infer_v1 import *

# causal direction inference for one dataset
# load a dataset
fpath = "./data/"
fname = 'pair0001.txt'
apath = fpath + fname
dir_groundtruth = [-1]  # 1: 'X->Y', -1: 'X<-Y'
data = np.loadtxt(apath)


# data processing (ds). 1: quantile norm; 2: in [0,1]; 3: zero mean and unit
# variance; 4: adaptive choice between 3 and 1 when many duplicate values appear
ds = 4
nvote = 9  # a direction wins by nvote votes if many duplicate values appear
quick = 1  # 1: the iteration is break if one direction wins with > half votes


direction = qbias_infer(fpath, 1, [1], dir_groundtruth,
						nvote, ds, quick, data)

# accuracy
mdir = direction[0, 3]
macc = direction[0, 4]
inf = "# direction infered: {:3.0f}, correctOrNot: {:.2f}".format(mdir, macc)
print(inf)
print(" - - -")

'''
# the result is saved to a file
bname = "./solution/dir_result.txt"
hds = "dataID Size groundTruth Prediction correctOrNot score_xy score_yx vote_xy0yx"
np.savetxt(bname, direction, fmt="%d	%d	%d	%d	%d	%8.5f	%8.5f	%d", header=hds)
'''
