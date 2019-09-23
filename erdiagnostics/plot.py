# -*- coding: utf-8 -*-
from .misc import wavefront_to_intensity
import numpy as np
import scipy.interpolate

def plot_relative_trajectories(reference_trajectory, names, trajectories, filename=None):
    import matplotlib.pyplot as plt
    z_min = max([np.min(trajectory[2]) for trajectory in trajectories])
    z_max = min([np.max(trajectory[2]) for trajectory in trajectories])
    mask = np.logical_and(z_min <= reference_trajectory[2], reference_trajectory[2] <= z_max)
    ref_x = reference_trajectory[0][mask]
    ref_y = reference_trajectory[1][mask]
    ref_z = reference_trajectory[2][mask]
    plt.subplot(211)
    plt.plot([z_min, z_max], [0, 0], label='reference')
    for name, trajectory in zip(names, trajectories):
        blah = 1000 * (scipy.interpolate.interp1d(trajectory[2], trajectory[0])(ref_z) - ref_x)
        if np.max(np.abs(blah)) < 0.001:
            plt.ylim(-0.001, 0.001)
        plt.plot(ref_z, blah, label=name)
    plt.xlabel('z (m)')
    plt.ylabel('x (mm)')
    plt.subplot(212)
    plt.plot([z_min, z_max], [0, 0], label='reference')
    for name, trajectory in zip(names, trajectories):
        blah = 1000 * (scipy.interpolate.interp1d(trajectory[2], trajectory[1])(ref_z) - ref_y)
        if np.max(np.abs(blah)) < 0.001:
            plt.ylim(-0.001, 0.001)
        plt.plot(ref_z, blah, label=name)
    plt.xlabel('z (m)')
    plt.ylabel('y (mm)')
    plt.tight_layout()
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    plt.clf()

def plot_intensity(intensity, detector_edge_length, plot_title='Intensity', filename=None):
    x_num, y_num = intensity.shape
    assert x_num == y_num
    detector_points = x_num
    import matplotlib.pyplot as plt
    x_min = -0.5 * detector_edge_length
    x_max = 0.5 * detector_edge_length
    x_num = detector_points
    y_min = -0.5 * detector_edge_length
    y_max = 0.5 * detector_edge_length
    y_num = detector_points
    x_step = (x_max - x_min) / (x_num - 1)
    y_step = (y_max - y_min) / (y_num - 1)
    x = np.linspace(x_min - 0.5 * x_step, x_max + 0.5 * x_step, x_num + 1)
    y = np.linspace(y_min - 0.5 * y_step, y_max + 0.5 * y_step, y_num + 1)
    plt.title(plot_title)
    plt.pcolormesh(1000 * x, 1000 * y, intensity, vmin=0, cmap='inferno')
    plt.xlim(1000 * x_min, 1000 * x_max)
    plt.ylim(1000 * y_min, 1000 * y_max)
    plt.colorbar().set_label('ph/s/.1%bw/mm^2')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
    plt.clf()
