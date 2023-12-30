# -*- coding: utf-8 -*-
# Create by Kaijun WANG in December 2023

import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.linear_model import LinearRegression


def norm1(x):
    xmin = min(x)
    xmax = max(x)
    if xmin == xmax:
        xnorm = np.ones(len(x), dtype=int)
        return xnorm
    xmax = xmax - xmin
    xnorm = (x - xmin) / xmax
    return xnorm


def data_std(xa):
    # Z-Score/standardization with the standard deviation
    xsd = np.std(xa, ddof=1)  # ddof=1: using /(N-1) instead of /N
    xmean = np.mean(xa)
    xastd = xa - xmean
    xastd[:] = xastd / xsd
    return xastd


def qscore(z, pred, tau):
    # quantile residual S(z,q,t)=(I(q>=z)-t)(q-z)
    nz = len(z)
    indx = np.zeros(nz, dtype=int)
    u = pred >= z
    indx[u] = 1
    qs = (indx - tau) * (pred - z)
    return qs


def dir_byscore(sxy, syx, win_small=1):
    # the direction is inferred by comparing sxy and syx
    cdj = 0
    # it is x->y if sxy is smaller
    if syx > sxy:
        cdj = 1
    elif syx < sxy:
        cdj = -1
    # it is x->y if sxy is bigger
    if win_small > 1:
        cdj = -cdj
    return (cdj)


def dir_voting(cdj, sxyj, syxj):
    # the direction by voting
    k1 = cdj == 1
    n1 = sum(k1)
    k2 = cdj == -1
    n2 = sum(k2)
    k3 = cdj == 0
    n3 = sum(k3)
    if n1 > n2:
        cd = 1
        k = k1
    elif n1 < n2:
        cd = -1
        k = k2
    else:
        cd = 0
        k = k1
    if n1 + n2 + n3 == n3:
        cd = 0
        k = k3
    score = np.zeros(3, dtype=np.float32)
    score[0] = np.mean(sxyj[k])
    score[1] = np.mean(syxj[k])
    score[2] = n1 * 100 + n2  # xy- yx-votes
    return cd, score


def allot_bins(xb, nx):
    # data are transformed to integers between 0 to nx, and allotted to bins
    # data are normalized to [0, 1]
    # nx = len(xb)
    xmin = min(xb)
    xmax = max(xb)
    if xmin == xmax:
        return 0, 0
    xmax = xmax - xmin
    xc = (xb - xmin) / xmax

    # every data value produces one integer, from 0 to nx
    m = nx
    if m < 30:
        m = 30
    xc[:] = np.round(m * xc) + 1
    xa = np.zeros(m, dtype=int)
    xa[:] = xc
    del xc

    # nx+1 bins/rows, each has nx columns; allot data points to bins
    m = m + 1
    bins = np.zeros(m, dtype=int)
    # every data point is allotted to a bin (row)
    for i in range(nx):  # index i is the sequence index of a data point
        u = xa[i] - 1  # the integer corresponding to a data point
        bins[u] = bins[u] + 1  # number of data points in bin u

    # record the bin serial-number of non-empty bins
    indxrow = np.zeros(m, dtype=int)
    j = -1
    for i in range(m):
        k = bins[i]  # number of data points in the ith bin
        if k >= 1:
            j = j + 1
            indxrow[j] = i + 1  # bin serial-number (1,2,3,...,nx+1)

    u = indxrow > 0  # for non-empty bins
    indxrow = indxrow[u] - 1  # bin serial-number (0,1,2,...,nx)
    bins = bins[indxrow]  # quantity in every non-empty bin
    n_bins = j + 1  # quantity of non-empty bins

    return bins, n_bins


def bins_divide(nbins, nrow, nfold=1):
    nb = round(np.sqrt(nrow) * nfold)  # total bin number
    if nb < 2:
        nb = 2

    nb2 = nb + 4
    id_bin = np.zeros(nb2, dtype=int)
    mb = int(np.floor(nrow / nb))
    nb1 = nb + 1
    for i in range(1, nb1):
        j1 = (i - 1) * mb
        j2 = j1 + mb
        id_bin[i] = np.sum(nbins[j1:j2])  # number of data points in a bin

    mc = np.round(mb / 2)
    id_bin2 = np.zeros(nb2, dtype=int)
    id_bin2[:] = id_bin
    kmin = 6
    ntail = nrow - nb * mb
    if ntail >= kmin and ntail > 0.7 * mb:
        id_bin[nb1] = np.sum(nbins[j2:nrow])
        id_bin2[nb] = id_bin[nb] + id_bin[nb1] - mc
    else:
        id_bin[nb] = np.sum(nbins[j1:nrow])
        nb1 = nb
        id_bin2[nb] = id_bin[nb] - mc

    id_bin2[1] = id_bin[1] + mc
    id_bin[0] = nb1  # 1st element is the size
    id_bin2[0] = nb
    return id_bin, id_bin2


def bin_merge(xbin, nwidth=2, cutail=0, n_bins=0):
    # nwidth bins are merged to 1 bin, the tail is merged by cutail*nwidth
    if n_bins == 0:
        n_bins = xbin[0]  # 1st element is the size
        nb2 = n_bins + 1

    nb = int(np.ceil(n_bins / nwidth))
    xbin_merge = np.zeros(nb + 2, dtype=int)
    xbin_merge[0] = nb
    nb1 = nb - 1
    if nwidth == 2:
        u2 = np.zeros(nb1, dtype=int)
        u2[:] = range(1, nb)
        u2 = 2 * u2  # 2 4 6
        u1 = u2 - 1  # 1 3 5
        xbin_merge[1:nb] = xbin[u1] + xbin[u2]
        j2 = u2[nb1 - 1]
    elif nwidth == 3:
        u3 = np.zeros(nb1, dtype=int)
        u3[:] = range(1, nb)
        u3 = 3 * u3  # 3 6
        u2 = u3 - 1  # 2 5
        u1 = u2 - 1  # 1 4
        xbin_merge[1:nb] = xbin[u1] + xbin[u2] + xbin[u3]
        j2 = u3[nb1 - 1]
    else:
        for i in range(1, nb):
            j1 = nwidth * (i - 1) + 1  # 1 3 5
            j2 = j1 + nwidth  # 3 5 7
            xbin_merge[i] = np.sum(xbin[j1:j2])

    j1 = j2 + 1
    xbin_merge[nb] = np.sum(xbin[j1:nb2])  # tail bin
    if cutail:
        ntail = n_bins - j1 + 1
        if ntail < cutail * nwidth:  # few in the tail bin
            xbin_merge[nb1] = xbin_merge[nb1] + xbin_merge[nb]
            xbin_merge[0] = nb - 1

    return (xbin_merge)


def foutlier(x, mp=1.5, sw=0):
    # output upper and lower results when sw>0, output len when sw=9
    q4 = np.quantile(x, 0.75)
    q2 = np.quantile(x, 0.25)
    iqr = q4 - q2
    uplimit = q4 + mp * iqr
    lowlimit = q2 - mp * iqr
    nx = len(x)
    u = x > uplimit  # upper part of x
    v = x < lowlimit  # lower part of x
    nu = np.sum(u)
    nv = np.sum(v)
    indx = np.zeros(nx, dtype=int)
    indx[:] = range(nx)
    if nu > 0 and nv > 0:
        w = np.concatenate((indx[u], indx[v]))  # axis=1
    elif nu > 0:
        w = indx[u]
    elif nv > 0:
        w = indx[v]
    else:
        w = -1

    if sw == 0:
        return w
    elif sw == 9:
        return (nu + nv)
    else:
        return [nu, nv]


def dup_count(x, y, cuts=3):
    # rate of duplicates, variation of ys at unique value of xs
    # it needs sorted xs for the duplicate count
    n = len(x)
    # the y range without outliers
    ye = foutlier(y, 3.0)
    w = np.setdiff1d(range(n), ye)
    ys = y[w]
    xs = x[w]
    dy = (np.max(ys) - np.min(ys)) / 2

    xs[:] = np.round(xs, cuts)
    # quantity of equal values, unique-value list, num of each unique value
    xuniq, x_num = np.unique(xs, return_counts=True)
    n_xuniq = len(xuniq)

    drate = np.zeros(3, dtype=np.float32)
    for i in range(1, 3):
        u = x_num >= (i + 1)
        k = x_num[u]
        n_max = len(k)
        if n_max == 0:
            break
        drate[i] = np.sum(k) / n  # quantity of duplicates to total n

    # variation of ys corresponding to the duplicates of a x-value is
    # measured by the deviation of ys when duplicate-quantity nk>=3
    n_max = max(x_num)
    ye = np.zeros(n_xuniq, dtype=np.float32)
    ye_std = np.zeros(n_xuniq, dtype=np.float32)
    nk = 3
    w = -1
    if n_max >= nk:
        j = 0
        for i in range(n_xuniq):
            k = x_num[i]
            u = range(j, (j + k))  # j:(j+k-1)
            if k > 1:
                ye[i] = np.sum(ys[u]) / k
                if k >= nk:
                    w = w + 1
                    # standard deviation of m ys for the x duplicates
                    ye_std[w] = np.std(ys[u], ddof=1) / dy
            else:
                ye[i] = ys[u]
            j = j + k

        w = w + 1
        ye_std = ye_std[0:w]
        ye_std[:] = np.sort(-ye_std)
        ye_std[:] = np.round(-ye_std, 4)

    # variation of ys is from the max part of standard deviations
    # no variation means: y values of x duplicates are same
    k = 1 - drate[2]
    if k < 0.2:
        k = 0.2  # 20% of w std, max 20%
    nk = k * w
    j = np.floor(nk) + 1
    if j > w:
        j = w
    j = int(j)
    drate[0] = np.mean(ye_std[0:j])
    drate = np.round(drate, 3)
    return drate


def dup_treat(mx, my):
    # mx[2]: rate of 3 x-duplicates, mx[0]: the deviation of y values
    b1 = mx[2] >= 0.40 and mx[0] >= 0.30  # large variation of y values
    b2 = mx[2] >= 0.60 and mx[0] >= 0.20  # 3-duplicate-quantity rate
    if b1 or b2:
        u1v = 1
    else:
        u1v = 0
    b1 = my[2] >= 0.40 and my[0] >= 0.30
    b2 = my[2] >= 0.60 and my[0] >= 0.20
    if b1 or b2:
        u2v = 1
    else:
        u2v = 0
    uv = u1v + u2v
    if uv > 1:
        uv = 1
    return uv


def qscore_bin(score_wt, indx, cheight, xbin, n_indx, nc=3, h=0):
    # output: bias between the scores in xbin and their mean
    # score_wt: [:,0] all, [:,1] up-half sum, [:,2] dw-half, [:,3-4] center
    # indx[:, 0:5]: all, up-half, dw-half, up-between, down-between
    sc_bias = np.zeros(10, dtype=np.float32)
    nbx = xbin[0]
    n_wt = 6
    sbin = np.zeros((nbx, n_wt), dtype=np.float32)
    jv = range(n_indx)  # [0, 1, 2, 3, 4]
    indx_mid = indx[:, 3] + indx[:, 4]
    j2 = 0
    for i in range(nbx):
        j1 = j2
        j2 = j2 + xbin[i + 1]
        ju = range(j1, j2)
        sk = np.sum(score_wt[ju, :] * indx[ju, :], axis=0)
        nk = np.sum(indx[ju, :], axis=0) + 0.00001
        sbin[i, jv] = sk / nk
        shalf = score_wt[ju, 4] * indx_mid[ju]  # score-mid, between up-dw
        nhalf = np.sum(indx_mid[ju]) + 0.00001
        sbin[i, 5] = np.sum(shalf) / nhalf

        if h > 0:
            height = np.mean(cheight[ju])
            if height < 0.01:
                height = np.median(cheight)
            sbin[i, :] = sbin[i, :] / height

    # sbin[:, 0:6]: all, up-half, dw-half, between up/down & center, between up & dw
    smean = np.mean(sbin, axis=0)
    for i in range(nbx):
        sbin[i, :] = np.abs(sbin[i, :] - smean)  # bias
    smean[:] = np.mean(sbin, axis=0)  # bias sum over regions / nbx
    # weighted scores
    sc_bias[0] = np.mean(smean[1:6])  # 4 and mid-between
    sc_bias[1] = (smean[1] + smean[2] + smean[3] + smean[4]) / 4  # up- dw- (half and between)
    sc_bias[2] = (smean[1] + smean[2] + 2 * smean[5]) / 4  # heavy mid-between
    sc_bias[3] = (smean[1] + smean[2] + smean[5]) / 3  # up- dw- and mid-between
    sc_bias[4] = (smean[1] + smean[2]) / 2  # up- dw-half

    return sc_bias


def regress_score(x, y, y_curves, x_bin, nx, nc=3):
    # scores based on quantile regression curves
    # data x, y and y_curves are in ascending order
    y = np.round(y, 7)
    y_curves = np.round(y_curves, 7)
    i_middle = int(np.floor(nc / 2))  # middle value
    res = np.zeros((nx, nc), dtype=np.float32)
    for i in range(nc):
        res[:, i] = y - y_curves[:, i]  # residuals between y and a curve
    yc_center = (y_curves[:, 0] + y_curves[:, nc - 1]) / 2  # central curve
    yc_center = np.round(yc_center, 7)

    # -------- data points over/below a curve ---------
    # assign data points to groups that over, below or between curves
    n_indx = nc + 2
    indx = np.zeros((nx, n_indx), dtype=int)
    indx[:, 0] = 1  # use the whole
    if nc > 1:
        temp = y_curves[:, i_middle]
        y_curves[:, i_middle] = yc_center
        k1 = nc - 1
        for i in [3, 4]:  # between 2 curves
            k2 = k1 - 1
            u = y >= y_curves[:, k2]
            v = y <= y_curves[:, k1]
            u[:] = u * v  # single and
            indx[u, i] = 1
            k1 = k2
    else:  # for nc==1
        indx[:, 3] = 1
        indx[:, 4] = 1

    u = y >= y_curves[:, i_middle]  # over/on the central curve
    indx[u, 1] = 1  # up
    u[:] = y <= y_curves[:, i_middle]  # below/on the central curve
    indx[u, 2] = 1  # dw
    if nc > 1:
        y_curves[:, i_middle] = temp
        del temp, u, v

    # the width between two curves
    width_c = np.abs(y_curves[:, nc - 1] - y_curves[:, 0])
    u = width_c < 0.01  # near 0
    k1 = np.sum(u)
    if k1 > 0:
        width_c[u] = np.median(width_c)
    del u

    # -------- weighted Quantile scores ---------
    if nc == 3:
        tau = [0.1127017, 0.5, 0.8872983]  # Quantile order/level
        wt = [0.2777778, 0.4444444, 0.2777778]  # weights
    else:
        tau = [0.5, 0.5]
        wt = [1, 1]

    res_c = y - yc_center
    del yc_center

    qc_wt = np.zeros((nx, 5), dtype=np.float32)  # all, up- dw-half
    qres = np.zeros((nx, nc), dtype=np.float32)
    for i in range(nc):
        qres[:, i] = qscore(res[:, i], 0, tau[i])
        if i == 0:  # qscore sum of nc weighted residuals
            q_wt = wt[i] * qres[:, i]
        else:
            q_wt[:] = q_wt + wt[i] * qres[:, i]

        if i == i_middle:
            qres[:, i] = qscore(res_c, 0, tau[i])
            res[:, i] = res_c

        if i == 0:
            qc_wt[:, 0] = wt[i] * qres[:, i]  # down-
        else:
            qc_wt[:, 0] = qc_wt[:, 0] + wt[i] * qres[:, i]  # central & up- qscore sum
        if i == i_middle:
            qc_wt[:, 2] = qc_wt[:, 0]  # down- & central score sum
            qc_wt[:, 1] = wt[i] * qres[:, i]  # central
        if i > i_middle:
            qc_wt[:, 1] = qc_wt[:, 1] + wt[i] * qres[:, i]  # central & up- qscore sum

    qmean = np.mean(q_wt)  # original weighted quantile score
    qc = np.mean(qc_wt[:, 0])  # central weighted quantile score
    wt_half = 0.7222222  # Consistency, 0.277+0.444
    # qwt_center = np.zeros((nx, 1), dtype=np.float32)[:, 0]
    qwt_center = qres[:, i_middle] * wt_half
    del res, q_wt, qres

    # qc_wt[:,0]: wt[0]*down + wt[1]*central + wt[2]*up, sum(wt)=1
    # qc_wt[:,1]: wt[1]*central + wt[2]*up  (up-half), sum(wt)=0.722
    # qc_wt[:,2]: wt[0]*down + wt[1]*central  (down-half), sum(wt)=0.722
    qc_wt[:, 3] = qwt_center
    qc_wt[:, 4] = qwt_center
    res_c[:] = np.abs(res_c)  # y - yc_center
    qheight = qscore_bin(qc_wt, indx, width_c, x_bin, n_indx, nc, 1)
    qc_bias = qscore_bin(qc_wt, indx, width_c, x_bin, n_indx, nc)
    qheight[5:10] = qc_bias[0:5]
    qheight[3] = qc
    qheight[4] = qmean
    return qheight


def regress_sim(x, y, xbin, nc, nx, curl_degree=0, dupturn=0):
    # output: simulated curves made by xbins
    # x needs to be sorted data
    if nc == 3:
        tau = [0.1127017, 0.5, 0.8872983]  # Quantile order/level
    else:
        tau = [0.5, 0.5]
    nbx = xbin[0]
    k = nbx + 1

    scurve = np.zeros((nx, nc), dtype=np.float32)
    x = x.reshape(-1, 1)
    j2 = 0
    for i in range(nbx):
        j1 = j2
        j2 = j2 + xbin[i + 1]
        ju = range(j1, j2)
        for k in range(nc):
            # Linear regression model that predicts conditional quantiles
            model = QuantileRegressor(quantile=tau[k], alpha=0, solver="highs")
            scurve[ju, k] = model.fit(x[ju, :], y[ju]).predict(x[ju, :])

    if dupturn:
        if curl_degree < 0.50:
            scurve[2:(nx - 1), :] = (scurve[3:nx, :] + scurve[2:(nx - 1), :]
                                     + scurve[1:(nx - 2), :] + scurve[0:(nx - 3), :]) / 4
    else:
        if curl_degree < 0.50:
            scurve[1:(nx - 1), :] = (scurve[2:nx, :] + scurve[1:(nx - 1), :]
                                     + scurve[0:(nx - 2), :]) / 3

    return scurve


def regress_lbin(x, y, xbin, nx):
    # output: a regression line made by xbins
    # x needs to be sorted data
    nbx = xbin[0]
    k = nbx + 1
    sline = np.zeros(nx, dtype=np.float32)
    x = x.reshape(-1, 1)
    j2 = 0
    for i in range(nbx):
        j1 = j2
        j2 = j2 + xbin[i + 1]
        ju = range(j1, j2)
        model = LinearRegression()
        model.fit(x[ju, :], y[ju])
        sline[ju] = model.predict(x[ju, :])

    return (sline)


def curl_probe(x, y, nbins, nrow, nx):
    # merge bins, produce the regression line for the merged bins
    n1 = nrow + 1
    n_rep = int(np.ceil(np.log2(nrow)))
    x_bin7 = np.zeros(n1, dtype=int)
    x_bin7[1:n1] = nbins
    x_bin7[0] = nrow
    n7 = nrow
    abreak = 0
    kmin = np.sqrt(nrow) * 1.3

    for i in range(n_rep):
        x_bin8 = x_bin7  # old bin
        x_bin7 = bin_merge(x_bin7)  # 2 bins are merged to one
        n7 = x_bin7[0]

        if n7 < 7:
            abreak = 1
        elif n7 < 15 and n7 < kmin:
            abreak = 1
        elif n7 < 17:  # decrease two by merging
            x_bin7[2] = x_bin7[1] + x_bin7[2]  # merge the head
            j2 = n7 - 1
            x_bin7[j2] = x_bin7[n7] + x_bin7[j2]  # merge the tail
            x_bin7[1:j2] = x_bin7[2:n7]
            n7 = n7 - 2
            x_bin7[0] = n7
            abreak = 1
        elif n7 < 19:
            x_bin7 = bin_merge(x_bin8, 3, 0.6)  # 3 bins are merged to one
            n7 = x_bin7[0]
            abreak = 1

        if abreak:
            j2 = n7 + 1
            n2 = np.median(x_bin7[1:j2])
            if x_bin7[n7] < 0.5 * n2:
                j1 = n7 - 1
                x_bin7[j1] = x_bin7[n7] + x_bin7[j1]
                j2 = n7
                n7 = n7 - 1
                x_bin7[0] = n7
            n1 = min(x_bin7[1:j2])
            if n1 > 2:
                x_bin7[j2] = 0
                break
            abreak = 0

    if n7 >= 4:
        x_bin5 = bin_merge(x_bin7)
    else:
        x_bin5 = x_bin7

    yline7 = regress_lbin(x, y, x_bin7, nx)
    if n7 > 3:
        j1 = round(nrow / 3)
        n1 = np.sum(nbins[0:j1])
        j2 = round(2 * nrow / 3)
        n2 = np.sum(nbins[j1:j2])
        n3 = nx - n1 - n2
        x_bin3 = [3, n1, n2, n3]
    else:
        x_bin3 = [1, nx]

    yline3 = regress_lbin(x, y, x_bin3, nx)
    ydif3 = np.abs(yline3 - yline7)
    j1 = round(nrow / 2)
    n2 = np.sum(nbins[0:j1])
    n3 = nx - n2
    x_bin2 = [2, n2, n3]
    yline2 = regress_lbin(x, y, x_bin2, nx)
    ydif2 = np.abs(yline2 - yline7)

    n3 = x_bin5[0]
    x_bin5[n3 + 1] = 0
    n2 = round(n3 / 2)
    if n2 < 1:
        n2 = 1
    y_bin2 = np.zeros(n3, dtype=np.float32)
    y_bin3 = np.zeros(n3, dtype=np.float32)
    j2 = 0
    for i in range(n3):
        j1 = j2
        j2 = j2 + x_bin5[i + 1]
        y_bin2[i] = np.mean(ydif2[j1:j2])
        y_bin3[i] = np.mean(ydif3[j1:j2])

    y_bin2[:] = - np.sort(-y_bin2)
    y_bin3[:] = - np.sort(-y_bin3)
    dif2 = np.mean(y_bin2[0:n2])
    dif3 = np.mean(y_bin3[0:n2])
    if dif2 > dif3:
        # curl degree, the base 0.899863 is from that of sin()
        r_curl = dif2 / 0.899863
    else:
        r_curl = dif3 / 0.899863
    if r_curl < 0.40:
        binout = x_bin5  # small bin
    else:
        binout = x_bin7  # big bin

    return r_curl, binout, x_bin7


def var_heter_bin(y_curves, xbin, nc):
    # the width between two curves
    width_c = np.abs(y_curves[:, nc - 1] - y_curves[:, 0])
    u = width_c < 0.01  # near 0
    k1 = np.sum(u)
    if k1 > 0:
        width_c[u] = np.median(width_c)
    nb = xbin[0]
    yc_bin = np.zeros(nb, dtype=np.float32)
    j2 = 0
    for i in range(nb):
        k = i + 1
        j1 = j2
        j2 = j2 + xbin[k]
        yc_bin[i] = np.mean(width_c[j1:j2])

    yc_bin[:] = np.sort(yc_bin)
    # heteroskedasticity: max 0-30% to 70-100%
    j2 = round(nb * 0.30) + 1
    k1 = np.mean(yc_bin[0:j2])
    j1 = nb - j2
    k = np.mean(yc_bin[j1:nb])
    rate_heter = k / k1

    return rate_heter


def rank_ties(x, fixed_seed, n):
    # breaking ties for equal values, using the random rank instead of the same rank
    if fixed_seed:
        np.random.seed(1)
    # n = len(x)
    # random values for random ranks
    w = np.random.uniform(0, 1, n)
    idx1 = np.argsort(x)
    x1 = x[idx1]

    # quantity of equal values, unique-value list, num of each unique value
    xuniq, x_num = np.unique(x1, return_counts=True)
    n1 = len(xuniq)
    idn = np.zeros(n1, dtype=int)
    idn[:] = range(n1)
    u = x_num > 1  # equal values (or duplicates)
    idn = idn[u]
    n2 = len(idn)

    for i in range(n2):
        j = idn[i]
        j1 = np.sum(x_num[0:j])
        j2 = j1 + x_num[j]  # quantity of duplicates
        u = idx1[j1: j2]
        # sorting random values for the ranks of equal values
        v = np.argsort(w[j1: j2])
        idx1[j1: j2] = u[v]

    idx2 = np.argsort(idx1)
    x2 = (idx2 + 1) / (n + 1)
    return x2


'''
'''