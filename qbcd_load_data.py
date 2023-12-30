# -*- coding: utf-8 -*-
# Create by Kaijun WANG in December 2023
import numpy as np
import pandas as pd

def data_meta(fpath, id_sim, apair=0):
    # the ground truth of directions for pairs
    if id_sim == 0:
        fname = fpath + f'README_polished.tab'  # 108 datasets
        odir = pd.read_table(fname, header=None, delimiter='\t')
        odir = odir.iloc[:, 5]
        osize = len(odir)
        directs = np.zeros(osize, dtype=int)
        directs[odir == '->'] = 1
        directs[odir == '<-'] = -1
        if apair:  # weights
            fname = fpath + f'pairmeta.txt'  # 108 rows with directions & weights
            odir = pd.read_table(fname, header=None, delim_whitespace=True)
            odir = odir.iloc[:, 5]
            weight = np.array(odir)
    elif id_sim == 1:  # SIM pairs
        apath = fpath + "/pairmeta.txt"
        odir = np.loadtxt(apath, delimiter=' ')
        osize = len(odir[:, 1])
        directs = np.zeros(osize, dtype=int)
        directs[:] = odir[:, 1]
        directs[directs == 2] = -1
    elif id_sim == 2:  # ANLSMN pairs
        apath = fpath + "/pairs_gt.txt"
        odir = np.loadtxt(apath)
        osize = len(odir)
        directs = np.zeros(osize, dtype=int)
        directs[:] = odir
        directs[directs == 0] = -1

    if id_sim > 0:
        weight = np.ones(osize, dtype=np.float32)
        weight[:] = 1 / osize
    return directs, weight


def load_adata(fpath, id_sim, k):
    # load a dataset
    if id_sim == 0:  # Tuebingen pairs
        apath = fpath + "/pair{:04d}.txt".format(k)
        data = np.loadtxt(apath)
    elif id_sim == 1:  # SIM pairs
        apath = fpath + "/pair{:04d}.txt".format(k)
        data = np.loadtxt(apath)
    elif id_sim == 2:  # ANLSMN pairs
        apath = fpath + "/pair_{:d}.txt".format(k)
        data = np.loadtxt(apath, delimiter=",", usecols=(1, 2), skiprows=1)

    return data
