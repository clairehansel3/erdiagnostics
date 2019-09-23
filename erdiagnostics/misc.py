# -*- coding: utf-8 -*-
import numpy as np
import scipy.interpolate
import srwlib

def track_drift(trajectory):
    dummy_field = srwlib.SRWLMagFldM(0, 1, 'n', 0.1, 0, 0)
    container = srwlib.SRWLMagFldC()
    container.add(dummy_field, 0, 0, 0.05)
    return srwlib.srwl.CalcPartTraj(trajectory, container, [1])

def rotate(x, y, angle):
    x_rot = x * np.cos(angle) - y * np.sin(angle)
    y_rot = x * np.sin(angle) + y * np.cos(angle)
    return x_rot, y_rot

def trajectory_to_array(trajectory):
    """
    Takes a srwlib.SRWLPrtTrj instance and converts it into a numpy array.
    """
    return np.array([getattr(trajectory, name) for name in ('arX', 'arY', 'arZ', 'arXp', 'arYp', 'arZp')])

def wavefront_to_intensity(wavefront):
    """
    Computes the intensity from a wavefront.
    """
    intensity = srwlib.array('f', [0] * wavefront.mesh.nx * wavefront.mesh.ny)
    srwlib.srwl.CalcIntFromElecField(intensity, wavefront, 6, 0, 3, wavefront.mesh.eStart, 0, 0)
    return np.array(intensity).reshape(wavefront.mesh.ny, wavefront.mesh.nx)

def interpolate_intensity(intensity, detector_edge_length_old, detector_edge_length_new, intensity_points_new='same as old'):
    assert detector_edge_length_new <= detector_edge_length_old
    x_num_old, y_num_old = intensity.shape
    x_num_new = x_num_old if intensity_points_new == 'same as old' else intensity_points_new
    y_num_new = y_num_old if intensity_points_new == 'same as old' else intensity_points_new
    x_old = np.linspace(-0.5 * detector_edge_length_old, 0.5 * detector_edge_length_old, x_num_old)
    y_old = np.linspace(-0.5 * detector_edge_length_old, 0.5 * detector_edge_length_old, y_num_old)
    x_new = np.linspace(-0.5 * detector_edge_length_new, 0.5 * detector_edge_length_new, x_num_new)
    y_new = np.linspace(-0.5 * detector_edge_length_new, 0.5 * detector_edge_length_new, y_num_new)
    return scipy.interpolate.interp2d(x_old, y_old, intensity)(x_new, y_new)

def compute_overlap(wavefront_1, wavefront_2):
    assert wavefront_dipole_1.mesh.xStart == wavefront_dipole_2.mesh.xStart
    assert wavefront_dipole_1.mesh.xFin == wavefront_dipole_2.mesh.xFin
    assert wavefront_dipole_1.mesh.nx == wavefront_dipole_2.mesh.nx
    assert wavefront_dipole_1.mesh.yStart == wavefront_dipole_2.mesh.yStart
    assert wavefront_dipole_1.mesh.yFin == wavefront_dipole_2.mesh.yFin
    assert wavefront_dipole_1.mesh.ny == wavefront_dipole_2.mesh.ny
    assert wavefront_dipole_1.mesh.eStart == wavefront_dipole_2.mesh.eStart
    # get mesh parameters
    x_min = wavefront_dipole_1.mesh.xStart
    x_max = wavefront_dipole_1.mesh.xFin
    x_num = wavefront_dipole_1.mesh.nx
    y_min = wavefront_dipole_1.mesh.yStart
    y_max = wavefront_dipole_1.mesh.yFin
    y_num = wavefront_dipole_1.mesh.ny
    intensity_overlap = np.multiply(intensity_1, intensity_2)
    # compute step size
    x_step = (x_max - x_min) / (x_num - 1)
    y_step = (x_max - x_min) / (y_num - 1)
    # compute integral using 2d trapezoid method
    return (1000 * x_step) * (1000 * y_step) * (
        # corners
        0.25 * (intensity_overlap[0, 0] + intensity_overlap[0, -1] + intensity_overlap[-1, 0] + intensity_overlap[-1, -1])
        # edges
        + 0.5 * (np.sum(intensity_overlap[0, 1:-1]) + np.sum(intensity_overlap[-1, 1:-1]) + np.sum(intensity_overlap[1:-1, 0]) + np.sum(intensity_overlap[1:-1, -1]))
        # center
        + np.sum(intensity_overlap[1:-1, 1:-1])
    )

def compute_overlap_intensity(intensity_1, intensity_2, detector_edge_length):
    x_num, y_num = intensity_1.shape
    intensity_overlap = np.multiply(intensity_1, intensity_2)
    # compute step size
    x_step = detector_edge_length / (x_num - 1)
    y_step = detector_edge_length / (y_num - 1)
    # compute integral using 2d trapezoid method
    return (1000 * x_step) * (1000 * y_step) * (
        # corners
        0.25 * (intensity_overlap[0, 0] + intensity_overlap[0, -1] + intensity_overlap[-1, 0] + intensity_overlap[-1, -1])
        # edges
        + 0.5 * (np.sum(intensity_overlap[0, 1:-1]) + np.sum(intensity_overlap[-1, 1:-1]) + np.sum(intensity_overlap[1:-1, 0]) + np.sum(intensity_overlap[1:-1, -1]))
        # center
        + np.sum(intensity_overlap[1:-1, 1:-1])
    )
