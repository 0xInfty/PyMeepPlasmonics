#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 12:32:57 2021

@author: vall
"""

import numpy as np
import matplotlib.pyplot as plt
import v_utilities as vu

#%%

resolution_wlen = list( range(1,321) )
resolution_test = [10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 130, 150,
                   170, 200, 220, 240, 260, 280, 300, 320]
resolution_index = [resolution_wlen.index(value) for value in resolution_test]

resolution_wlen = np.array(resolution_wlen)
wlen = np.array( [1]*len(resolution_wlen) )
resolution = resolution_wlen
from_um_factor = np.array( [1e3]*len(resolution_wlen) )
time_period_factor = np.array( [10]*len(resolution_wlen) )
n_period_line = np.array( [100]*len(resolution_wlen) )
n_period_plane = np.array( [100]*len(resolution_wlen) )
courant = np.array( [0.5]*len(resolution_wlen) )

until_time = time_period_factor * wlen

period_line = wlen / n_period_line
period_plane = wlen / n_period_plane

minimum_division_space = 1 / resolution
minimum_division_time = courant / resolution

t_points_line = until_time / period_line

round_up_until_time = np.array([vu.round_to_multiple(until_time[k], courant[k]/resolution[k], round_up=True) for k in range(len(resolution_wlen))]) # chosen
round_down_until_time = np.array([vu.round_to_multiple(until_time[k], courant[k]/resolution[k], round_down=True) for k in range(len(resolution_wlen))])
round_until_time = np.array([vu.round_to_multiple(until_time[k], courant[k]/resolution[k]) for k in range(len(resolution_wlen))]) 

round_up_period_line = np.array([vu.round_to_multiple(period_line[k], courant[k]/resolution[k], round_up=True) for k in range(len(resolution_wlen))])
round_down_period_line = np.array([vu.round_to_multiple(period_line[k], courant[k]/resolution[k], round_down=True) for k in range(len(resolution_wlen))]) # chosen
round_period_line = np.array([vu.round_to_multiple(period_line[k], courant[k]/resolution[k]) for k in range(len(resolution_wlen))])

round_down_period_plane = np.array([vu.round_to_multiple(period_plane[k], courant[k]/resolution[k], round_down=True) for k in range(len(resolution_wlen))])

round_up_t_points_line = np.round( round_up_until_time / round_down_period_line ) # chosen
round_down_t_points_line = np.round( round_down_until_time / round_up_period_line )
round_t_points_line = np.round( round_down_until_time / round_up_period_line )

#%%

fig, axes = plt.subplots(nrows=3, sharex=True, gridspec_kw={"hspace":0})

l1, = axes[0].plot(resolution_wlen, round_until_time, "oC0", alpha=0.3, markersize=8, label="Round")
l2, = axes[0].plot(resolution_wlen, round_up_until_time, "oC1", alpha=0.3, markersize=8, label="Round up")
l3, = axes[0].plot(resolution_wlen, round_down_until_time, "oC2", alpha=0.3, markersize=8, label="Round down")
l4, = axes[0].plot(resolution_wlen[resolution_index], round_up_until_time[resolution_index], "x", color="k", markersize=10, label="Chosen")
l0, = axes[0].plot(resolution_wlen, until_time, "-k", label="Value")
axes[0].set_ylabel("Until time [MPu]")
axes[0].grid()

lines = [l0, l1, l2, l3, l4]
axes[0].legend(lines, [l.get_label() for l in lines])

n1, = axes[1].plot(resolution_wlen, round_period_line, "oC0", alpha=0.3, markersize=8, label="Round")
n2, = axes[1].plot(resolution_wlen, round_up_period_line, "oC1", alpha=0.3, markersize=8, label="Round up")
n3, = axes[1].plot(resolution_wlen, round_down_period_line, "oC2", alpha=0.3, markersize=8, label="Round down")
n5, = axes[1].plot(resolution_wlen, minimum_division_time, "k", linestyle="dashed", label=r"Ref. $\Delta t$")
n4, = axes[1].plot(resolution_wlen[resolution_index], round_down_period_line[resolution_index], "x", color="k", markersize=10, label="Chosen")
n0, = axes[1].plot(resolution_wlen, period_line, "-k", label="Value")
axes[1].set_ylabel("Period line [MPu]")
axes[1].grid()

new_lines = [n0, n1, n2, n3, n4, n5]
axes[1].legend(new_lines, [n.get_label() for n in new_lines])

t1, = axes[2].plot(resolution_wlen, round_t_points_line, "oC0", alpha=0.3, markersize=8, label="Round")
t2, = axes[2].plot(resolution_wlen, round_up_t_points_line, "oC1", alpha=0.3, markersize=8, label="Round up")
t3, = axes[2].plot(resolution_wlen, round_down_t_points_line, "oC2", alpha=0.3, markersize=8, label="Round down")
t4, = axes[2].plot(resolution_wlen[resolution_index], round_up_t_points_line[resolution_index], "x", color="k", markersize=10, label="Chosen")
t0, = axes[2].plot(resolution_wlen, t_points_line, "-k", label="Value")
axes[2].set_ylabel("Number of points in time")
axes[2].grid()

last_lines = [t0, t1, t2, t3, t4]
axes[2].legend(last_lines, [t.get_label() for t in last_lines])

axes[-1].set_xlabel(r"Resolution [points/$\lambda$]")
plt.tight_layout()
