# -*- coding: utf-8 -*-
# Create by Kaijun WANG in December 2023

from qbcd_bias_infer_v1 import *
from qbcd_load_data import data_meta

path_prefix = '.'  # or the absolute path such as "E:/wkjwang/code/causal"
if 1:
  if 1:
    isim = 1
    exts = ["SIM", "SIM-c", "SIM-ln", "SIM-G"]
    path = "/data/Benchmark_simulated/"
  else:
    isim = 2
    exts = ["AN", "AN-s", "LS", "LS-s", "MN-U"]
    path = "/data/ANLSMN_pairs/"

  # select a dataset to run
  id_set = 1
  ext = exts[id_set - 1]
  fpath = path_prefix + path + ext
  dir_groundtruth, weight = data_meta(fpath, isim)
  n_pairs = 100
  pairs = np.array(range(n_pairs)) + 1
  weight = np.ones(n_pairs, int)
else:
  isim = 0
  path = "/data/causal_tuebingen/"
  fpath = path_prefix + path
  ext = "Tuebingen"
  pairs = np.array(range(1, 109))
  nopair = [47, 52, 53, 54, 55, 70, 71, 105, 107]
  pairs = np.setdiff1d(pairs, nopair)  # 99 data sets
  n_pairs = 99
  # groundtruth directions and weights of 108 data sets
  dir_groundtruth, weight = data_meta(fpath, isim, 1)


# data processing (ds). 1: quantile norm; 2: in [0,1]; 3: zero mean and unit
# variance; 4: adaptive choice between 3 and 1 when many duplicate values appear
ds = 4
nvote = 9  # a direction wins by nvote votes if many duplicate values appear
quick = 1  # 1: the iteration is break if one direction wins with > half votes

wsums = 0
acc_ws = 0
kmax = int(n_pairs / 10)
kleft = n_pairs - kmax * 10
if kleft > 0:
   kmax = kmax + 1
k = kmax
ks = np.array(range(10))
while k:  # all the data sets are divided into k groups to run
  u = (kmax - k) * 10 + ks
  if k == 1 and kleft > 0:
    u = u[0:kleft]
  pair2 = pairs[u]

  direction = qbias_infer(fpath, isim, pair2, dir_groundtruth,
                          nvote, ds, quick)
  pair1 = pair2 - 1
  m = len(pair1)
  wsums = wsums + sum(weight[pair1])  # weight sum
  acc_ws = acc_ws + sum(direction[0:m, 4] * weight[pair1])
  nr = 0
  if k < kmax:
    db = np.loadtxt(bname, delimiter="\t", skiprows=1)
    nr = len(db[:, 0])
  elif k == kmax:  # the result is saved to a file
    bname = "./solution/dir_result.txt"

  if nr > 1:
    nr = nr - 1
    db = np.array(db)
    direction[m, :] = direction[m, :] + db[nr, :]
    direction[m, 5] = acc_ws / wsums
    direction[m, 1] = 1000 * direction[m, 4] / n_pairs
    db = np.vstack((db[0:nr, :], direction))
  else:
    db = direction
  hds = "dataID Size groundTruth Prediction correctOrNot score_xy score_yx vote_xy0yx"
  np.savetxt(bname, db, fmt="%d	%d	%d	%d	%d	%8.5f	%8.5f	%d", header=hds)

  # average accuracy
  k = k - 1
  if k == 0:
    macc1 = direction[m, 4] / n_pairs
    macc2 = direction[m, 1] / 1000
    inf = "# {} dataset-group accuracy: {:.4f}, weighted accuracy:{:.4f}". \
      format(ext, macc1, macc2)
    print(inf)
    print(" - - - ")
