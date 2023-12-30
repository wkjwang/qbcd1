# -*- coding: utf-8 -*-
# Create by Kaijun WANG in December 2023

import timeit
from scipy.stats import norm
from qbcd_utils import *
from qbcd_load_data import load_adata


def qbias_infer(fpath, isim, pair_id, ref_dir,
                nvote, ds, quick=0, data=[0]):
    # nvote: solutions from nvote iterations
    n_qt = 3  # number of quantiles
    n_dataset = len(pair_id)
    n_size = n_dataset + 1
    fixed_seed = 1
    ninput_data = len(data)

    n_corr = 0
    n_iter = 0
    corr1 = np.zeros(n_size, int)
    corr2 = np.zeros(n_size, int)
    corr3 = np.zeros(n_size, int)
    corr4 = np.zeros(n_size, int)
    corr5 = np.zeros(n_size, int)
    corr6 = np.zeros(n_size, int)
    corr11 = np.zeros(n_size, int)
    score_xy = np.zeros(n_size, dtype=np.float32)
    score_xy1 = np.zeros(n_size, dtype=np.float32)
    score_xy2 = np.zeros(n_size, dtype=np.float32)
    score_xy3 = np.zeros(n_size, dtype=np.float32)
    score_xy4 = np.zeros(n_size, dtype=np.float32)
    score_xy5 = np.zeros(n_size, dtype=np.float32)
    score_xy6 = np.zeros(n_size, dtype=np.float32)
    score_yx = np.zeros(n_size, dtype=np.float32)
    score_yx1 = np.zeros(n_size, dtype=np.float32)
    score_yx2 = np.zeros(n_size, dtype=np.float32)
    score_yx3 = np.zeros(n_size, dtype=np.float32)
    score_yx4 = np.zeros(n_size, dtype=np.float32)
    score_yx5 = np.zeros(n_size, dtype=np.float32)
    score_yx6 = np.zeros(n_size, dtype=np.float32)
    binf3 = np.zeros(n_size, dtype=np.float32)
    dupr2 = np.zeros(n_size, dtype=np.float32)
    n_data = np.zeros(n_size, dtype=int)
    pred_dir = np.zeros(n_size, dtype=int)
    vote_xyx = np.zeros(n_size, dtype=int)
    runtime = np.zeros(n_size, dtype=np.float32)
    f_nn = np.zeros(n_size, dtype=np.float32)
    curl_degree = np.zeros(n_size, dtype=np.float32)
    pair_ids = np.zeros(n_size, dtype=int)
    pair_ids[0:n_dataset] = pair_id
    dir_true = np.zeros(n_size, dtype=int)
    nhalf = nvote / 2
    n_size = n_dataset

    for i in range(n_dataset):
        # data load & preprocessing
        t1 = timeit.default_timer()
        da_id = pair_id[i]
        dir_true[i] = ref_dir[da_id - 1]
        if ninput_data == 1:
            data = load_adata(fpath, isim, da_id)
        if ds == 2:
            data[:, 0] = norm1(data[:, 0])
            data[:, 1] = norm1(data[:, 1])
        elif ds >= 3 or ds == 1:
            data[:, 0] = data_std(data[:, 0])
            data[:, 1] = data_std(data[:, 1])

        nx = len(data[:, 0])
        n_data[i] = nx
        idx = np.argsort(data[:, 0])
        idy = np.argsort(data[:, 1])
        # using float32 and 7 decimal places for data
        xsort = np.zeros((nx, 2), dtype=np.float32)
        ysort = np.zeros((nx, 2), dtype=np.float32)
        xsort[:, ] = np.round(data[idx,], 7)
        ysort[:, ] = np.round(data[idy,], 7)
        ox = dup_count(xsort[:, 0], xsort[:, 1])
        oy = dup_count(ysort[:, 1], ysort[:, 0])
        # duplicate index, 1: x/y has heavy duplicates, 2: both
        dupr2[i] = dup_treat(ox, oy)
        noncurlturn = 0

        # adaptive choice when many duplicate values
        if (ds == 4 and dupr2[i] >= 1) or ds == 1:
            ds_i = 1
            vote_a = nvote
            dupturn = 1
            data[:, 0] = xsort[:, 0]
            data[:, 1] = xsort[:, 1]
        else:
            ds_i = ds
            vote_a = 1
            dupturn = 0
            del data, idx, idy

        cdj1 = np.zeros(vote_a, dtype=int)
        cdj2 = np.zeros(vote_a, dtype=int)
        cdj3 = np.zeros(vote_a, dtype=int)
        cdj4 = np.zeros(vote_a, dtype=int)
        cdj5 = np.zeros(vote_a, dtype=int)
        cdj6 = np.zeros(vote_a, dtype=int)
        sxyj1 = np.zeros(vote_a, dtype=np.float32)
        sxyj2 = np.zeros(vote_a, dtype=np.float32)
        sxyj3 = np.zeros(vote_a, dtype=np.float32)
        sxyj4 = np.zeros(vote_a, dtype=np.float32)
        sxyj5 = np.zeros(vote_a, dtype=np.float32)
        sxyj6 = np.zeros(vote_a, dtype=np.float32)
        syxj1 = np.zeros(vote_a, dtype=np.float32)
        syxj2 = np.zeros(vote_a, dtype=np.float32)
        syxj3 = np.zeros(vote_a, dtype=np.float32)
        syxj4 = np.zeros(vote_a, dtype=np.float32)
        syxj5 = np.zeros(vote_a, dtype=np.float32)
        syxj6 = np.zeros(vote_a, dtype=np.float32)
        direct = np.zeros(6, dtype=int)

        # ------ computing regression curves & scores ------
        nx1 = nx + 1
        h1 = nx < 250
        sxy = np.zeros(7, dtype=np.float32)  # x->y
        syx = np.zeros(7, dtype=np.float32)  # y->x
        sxyj = 0.0
        syxj = 0.0
        fnn = 101
        t2 = t1

        for j in range(vote_a):
            # ------ data are transformed to be normal distribution ------
            if ds_i == 1:
                # data values are transformed to ranks, and divided by nx+1
                dx = rank_ties(data[:, 0], fixed_seed, nx1)
                dy = rank_ties(data[:, 1], fixed_seed, nx1)
                # inverse quantile of Cumulative Distribution Function
                dx[:] = norm.ppf(dx)
                dy[:] = norm.ppf(dy)
                idx[:] = np.argsort(dx)
                xsort[:, 0] = dx[idx]
                xsort[:, 1] = dy[idx]
                idy[:] = np.argsort(dy)
                ysort[:, 0] = dx[idy]
                ysort[:, 1] = dy[idy]

            # ------ divide data into bins & probe curl ------
            ox_bin, ox_nr = allot_bins(xsort[:, 0], nx)
            oy_bin, oy_nr = allot_bins(ysort[:, 1], nx)
            if dupturn or h1:  # or small data quantity
                xbin, xbin2 = bins_divide(ox_bin, ox_nr)  # x numb in bins, 2: Half in bin1 &2
                ybin, ybin2 = bins_divide(oy_bin, oy_nr)
            ycur, ux_bin, ux_binbig = curl_probe(xsort[:, 0], xsort[:, 1], ox_bin, ox_nr, nx)
            xcur, uy_bin, uy_binbig = curl_probe(ysort[:, 1], ysort[:, 0], oy_bin, oy_nr, nx)
            del ox_bin, oy_bin
            if not dupturn:
                h2 = xcur < 0.4 or ycur < 0.4
                if not h1:  # default: not small data quantity
                    xbin = ux_bin
                    ybin = uy_bin
                elif h1 and h2:
                    xbin = ux_bin
                    ybin = uy_bin

            # ------ piecewise quantile-regression lines by bins ------
            ypj = regress_sim(xsort[:, 0], xsort[:, 1], xbin, n_qt, nx, ycur, dupturn)
            xpj = regress_sim(ysort[:, 1], ysort[:, 0], ybin, n_qt, nx, xcur, dupturn)

            # ---- scores based on quantile curves ----
            # 7 decimal places for data and ypj xpj in regress_score
            xy_qbias = regress_score(xsort[:, 0], xsort[:, 1], ypj, xbin, nx, n_qt)
            yx_qbias = regress_score(ysort[:, 1], ysort[:, 0], xpj, ybin, nx, n_qt)
            if dupturn:
                varh_xy = var_heter_bin(ypj, xbin, n_qt)
                varh_yx = var_heter_bin(xpj, ybin, n_qt)
            else:
                varh_xy = var_heter_bin(ypj, ux_binbig, n_qt)
                varh_yx = var_heter_bin(xpj, uy_binbig, n_qt)

            # store the bias scores
            sxy[0:2] = xy_qbias[0:2]
            syx[0:2] = yx_qbias[0:2]
            sxy[2:4] = xy_qbias[[5, 7]]
            syx[2:4] = yx_qbias[[5, 7]]
            sxy[4:6] = xy_qbias[3:5]
            syx[4:6] = yx_qbias[3:5]
            sxy = 100000 * sxy
            syx = 100000 * syx
            sxy[6] = varh_xy
            syx[6] = varh_yx

            # ------ direction judgement with scores ------
            # default: x->y if smaller sxy
            sxyj1[j] = sxy[0]
            syxj1[j] = syx[0]
            cdj1[j] = dir_byscore(sxyj1[j], syxj1[j])

            sxyj2[j] = sxy[1]
            syxj2[j] = syx[1]
            cdj2[j] = dir_byscore(sxyj2[j], syxj2[j])

            sxyj3[j] = sxy[2]
            syxj3[j] = syx[2]
            cdj3[j] = dir_byscore(sxyj3[j], syxj3[j])

            sxyj4[j] = sxy[3]
            syxj4[j] = syx[3]
            cdj4[j] = dir_byscore(sxyj4[j], syxj4[j])

            sxyj5[j] = sxy[4]  # original quantile score
            syxj5[j] = syx[4]
            cdj5[j] = dir_byscore(sxyj5[j], syxj5[j])

            sxyj6[j] = sxy[5]
            syxj6[j] = syx[5]
            cdj6[j] = dir_byscore(sxyj6[j], syxj6[j])

            sxyj = sxyj + sxy[6]
            syxj = syxj + syx[6]

            if n_data[i] > 10000:
                if ds_i == 1:
                    nk1 = cdj5[j]
                else:
                    nk1 = cdj1[j]
                t3 = timeit.default_timer()
                t4 = t3 - t2
                inf = '=runj= data {}, iteration {}, direct_pred {}, secends:{:.2f}'.format(da_id, j, nk1, t4)
                print(inf)
                t2 = t3

            # wins over nhalf iterations, break to save time
            if quick and ds_i == 1 and j > nhalf:
                ks = cdj5 == 1
                nk1 = sum(ks)
                ks = cdj5 == -1
                nk2 = sum(ks)
                if (nk1 > nhalf) or (nk2 > nhalf):
                    break
            # end of j

        # the direction is inferred by votes
        cd, ms = dir_voting(cdj1, sxyj1, syxj1)
        direct[0] = cd
        score_xy1[i] = ms[0]  # mean score
        score_yx1[i] = ms[1]
        vote1 = int(ms[2])

        # another record of solution
        cd, ms = dir_voting(cdj2, sxyj2, syxj2)
        direct[1] = cd
        score_xy2[i] = ms[0]
        score_yx2[i] = ms[1]

        cd, ms = dir_voting(cdj3, sxyj3, syxj3)
        direct[2] = cd
        score_xy3[i] = ms[0]
        score_yx3[i] = ms[1]

        cd, ms = dir_voting(cdj4, sxyj4, syxj4)
        direct[3] = cd
        score_xy4[i] = ms[0]
        score_yx4[i] = ms[1]

        cd, ms = dir_voting(cdj5, sxyj5, syxj5)
        direct[4] = cd
        score_xy5[i] = ms[0]
        score_yx5[i] = ms[1]
        vote2 = int(ms[2])

        cd, ms = dir_voting(cdj6, sxyj6, syxj6)
        direct[5] = cd
        score_xy6[i] = ms[0]
        score_yx6[i] = ms[1]

        # directions & accuracy for every vote
        corr1[i] = int(dir_true[i] == direct[0])
        corr2[i] = int(dir_true[i] == direct[1])
        corr3[i] = int(dir_true[i] == direct[2])
        corr4[i] = int(dir_true[i] == direct[3])
        corr5[i] = int(dir_true[i] == direct[4])
        corr6[i] = int(dir_true[i] == direct[5])

        # difference between mid-curve and ypmean, select the bigger one
        sxyj = sxyj / vote_a
        syxj = syxj / vote_a
        # record the heter variance
        hvar = (sxyj < 1.5) and (syxj < 1.5)
        k2 = syxj / 100
        if k2 >= 1:
            k2 = 0.999
        binf3[i] = round(100 * sxyj) + k2

        # record the curl degree
        hcurl = (ycur < 0.11) and (xcur < 0.11)
        ycur = round(ycur * 1000)
        xcur = round(xcur, 3)
        if xcur >= 1:
            xcur = 0.999
        curl_degree[i] = ycur + xcur

        # result choice by the solution votes
        if not dupturn and hcurl and hvar:
            noncurlturn = 1
        if dupturn or noncurlturn:
            pred_dir[i] = direct[4]  # 4:qc
            corr11[i] = corr5[i]
            vote_xyx[i] = vote2
            score_xy[i] = score_xy5[i]
            score_yx[i] = score_yx5[i]
        else:
            pred_dir[i] = direct[0]  # 0: first score
            corr11[i] = corr1[i]
            vote_xyx[i] = vote1
            score_xy[i] = score_xy1[i]
            score_yx[i] = score_yx1[i]

        f_nn[i] = fnn
        n_corr = n_corr + corr11[i]
        n_iter = n_iter + 1
        accs = 100 * round(n_corr / n_iter, 3)
        t5 = round((timeit.default_timer() - t1), 2)
        runtime[i] = t5
        if i % 10 == 9 or i == n_dataset or nx > 5000:
            inf = '={:3d} pair{}, Votexy0yx={}, correctN={}, acc%={:.1f}, time:{}'.format(i, da_id, vote_xyx[i], n_corr, accs, t5)
            print(inf)
        # end of i

    # ------ output collections ------
    score_xy1[:] = np.int32(score_xy1)
    score_xy2[:] = np.int32(score_xy2)
    score_xy3[:] = np.int32(score_xy3)
    score_xy4[:] = np.int32(score_xy4)
    score_xy5[:] = np.int32(score_xy5)
    score_xy6[:] = np.int32(score_xy6)
    score_xy[:] = np.int32(score_xy)
    score_yx1[:] = np.int32(score_yx1)
    score_yx2[:] = np.int32(score_yx2)
    score_yx3[:] = np.int32(score_yx3)
    score_yx4[:] = np.int32(score_yx4)
    score_yx5[:] = np.int32(score_yx5)
    score_yx6[:] = np.int32(score_yx6)
    score_yx[:] = np.int32(score_yx)
    k = i + 1
    corr1[k] = sum(corr1)
    corr2[k] = sum(corr2)
    corr3[k] = sum(corr3)
    corr4[k] = sum(corr4)
    corr5[k] = sum(corr5)
    corr6[k] = sum(corr6)
    corr11[k] = sum(corr11)
    runtime[k] = sum(runtime)
    pair_ids[k] = n_size

    k = k + 1
    sout = np.zeros((k, 8), dtype=np.float32)
    k = k - 1
    vote_xyx[k] = runtime[k]
    sout[:, 0] = pair_ids
    sout[:, 1] = n_data
    sout[:, 2] = dir_true
    sout[:, 3] = pred_dir
    sout[:, 4] = corr11
    sout[:, 5] = score_xy / 100000
    sout[:, 6] = score_yx / 100000
    sout[:, 7] = vote_xyx

    return sout
