# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 11:42:59 2018

@author: ynomura
"""
from __future__ import print_function

import numpy as np
# import scipy as sp
from scipy import signal

# Extraction of Lung ROI


def extract_lung_roi(img_data, pixel_spacing):

    row = img_data.shape[0]
    column = img_data.shape[1]

    # Step 1: Extraction of initial lung ROI
    #   (by Tsujii O, Med Phys 25;998-1007:1998)

    # (1-1) define x0(x_right), x_center, x1(x_left)
    p_y = np.sum(img_data, axis=0)

    K = np.int(np.round(10.0 / pixel_spacing[0]))  # also 10 mm
    if K % 2 == 0:
        K = K + 1

    k_half = np.int(np.floor(np.float(K) / 2.0))
    b = np.ones(K) / np.float(K)

    p_a = np.convolve(p_y, b, mode='same')
    p_d = np.gradient(p_a)

    p_m = np.convolve(p_d, b, mode='same')

    x_start = np.int(np.floor(0.3 * column))
    x_end = np.int(np.ceil(0.7 * column))

    x_max = np.argmax(p_m[x_start:x_end]) + x_start

    pm_minus = np.where(p_m[0:x_max] < 0)
    if pm_minus[0].size==0:
        x0 = 0
    else:
        x0 = pm_minus[0][pm_minus[0].size - 1]

    pm_minus = np.where(p_m[x_max:column - 1] < 0)
    if pm_minus[0].size==0:
        x_center = x_max
    else:
        x_center = pm_minus[0][0] + x_max

    x1 = x0 + 2 * (x_center - x0)

    # (1-2) define y0(y_top) and y1(y_bottom)
    R = np.int(np.round(25.0 / pixel_spacing[0]))

    s_x = np.sum(img_data[:, x0 - k_half:x0 + k_half], axis=1) / np.float(K)
    s_a = np.convolve(s_x, b, mode='same')

    s_d = np.gradient(s_a)

    y1_start = np.int(0.5 * np.floor(row))  # 0.5で良いのか要確認
    y1_end = np.int(0.9 * np.floor(row))
    y1 = np.argmax(s_d[y1_start:y1_end]) + y1_start

    y_center = y1 - np.int(1.5 * (x_center - x0))

    s_b = (s_a[0:y_center + R] < np.median(s_a[0:y_center])).astype('int')

    y0 = 0
    run_length_max = R

    for y_pos in range(y_center, 0, -1):
        tmp_val = sum(s_b[y_pos:y_pos + R])

        if tmp_val < run_length_max:
            run_length_max = tmp_val
            y0 = y_pos
            # print(run_length_max, y_top)

    # define xr1 and xl2 (y_top, y_bottom設定に必要)
    x_sum1 = np.sum(img_data[0:y_center, x0:x1], axis=0)
    x_thorax_th = np.max(x_sum1) * 0.9

    x_thorax_area = np.where(x_sum1 >= x_thorax_th)
    xr1 = x_thorax_area[0][0] + x0
    xl2 = x_thorax_area[0][x_thorax_area[0].size - 1] + x0

    # define xl1 and xr2
    x_start = np.int(np.floor(0.05 * np.float(column)))
    x_end = np.int(np.floor(0.95 * np.float(column)))
    x_sum2 = np.sum(img_data[y0:y1, :], axis=0)

    # 本当はこうしたい
    # x_peak = signal.argrelmax(x_sum2, order=K)
    # xl1 = x_peak[0][0] + x_start
    # xr2 = x_peak[0][x_peak[0].size-1] + x_start

    x_start = np.int(np.floor(0.05 * np.float(column)))
    x_end = np.int(np.floor(0.25 * np.float(column)))
    xl1 = np.argmax(x_sum2[x_start:x_end]) + x_start

    x_start = np.int(np.floor(0.75 * np.float(column)))
    x_end = np.int(np.floor(0.95 * np.float(column)))
    xr2 = np.argmax(x_sum2[x_start:x_end]) + x_start

     # define y_top
    yt_sum1 = np.sum(img_data[:, xl1 + np.int((xr1 - xl1) / 2.0):xr1], axis=1)
    yt_sum2 = np.sum(img_data[:, xl2:xr2 - np.int((xr2 - xl2) / 2.0)], axis=1)

    yt_start = np.int(np.floor(0.01 * np.float(row)))
    peak_id = signal.argrelmax(yt_sum1[yt_start:y_center], order=30)
    if len(peak_id[0]) == 0:
        yt_right = y_center
    else:
        yt_right = peak_id[0][0] + yt_start

    peak_id = signal.argrelmax(yt_sum2[yt_start:y_center], order=30)
    if len(peak_id[0]) == 0:
        yt_left = y_center
    else:
        yt_left = peak_id[0][0] + yt_start

    y_top = np.min([yt_right, yt_left, y0])

    print(yt_right, yt_left, y0)

    # define y_bottom
    yb_sum1 = np.sum(img_data[:, xl1:xl1 + np.int((xr1 - xl1) / 4.0)], axis=1)
    yb_sum1 = np.gradient(yb_sum1)
    yb_sum1 = np.convolve(yb_sum1, b, mode='same')

    y_end = np.int(np.floor(0.9 * np.float(row)))
    yb_right = np.argmax(yb_sum1[y1:y_end]) + y1

    yb_sum2 = np.sum(img_data[:, xr2 - np.int((xr2 - xl2) / 4.0):xr2], axis=1)
    yb_sum2 = np.gradient(yb_sum2)
    yb_sum2 = np.convolve(yb_sum2, b, mode='same')

    yb_left = np.argmax(yb_sum2[y1:y_end]) + y1

    y_bottom = np.max([yb_right, yb_left, y1])
    print(yb_right, yb_left, y1)

    return np.array([[x0, x1, y0, y1], [xl1, xr2, y_top, y_bottom]])


def signal_normalization(img_data, lung_roi):

    cropped_img = img_data[lung_roi[2]:lung_roi[3], lung_roi[0]:lung_roi[1]]

    myu = np.average(cropped_img)
    sigma = np.std(cropped_img)

    print(myu, sigma)

    return ((img_data.astype('float') - myu) / sigma)
