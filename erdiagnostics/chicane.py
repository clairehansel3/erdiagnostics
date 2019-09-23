# -*- coding: utf-8 -*-
from .dogleg import Dogleg
from .misc import trajectory_to_array, rotate
import numpy as np
import scipy.interpolate

class Chicane(object):
    """
    A Chicane consists of four bending magnets and three detectors. Internally
    it consists of three doglegs. Be careful about changing attribututes of each
    section: for example if you change chicane.section_1.dipole_2_strength you
    MUST change chicane.section_2.dipole_1_strength because they represent the
    same dipole.

    Attributes:
    * section_1
    * section_2
    * section_3
    * beginning_x
    * beginning_y
    * beginning_bx
    * beginning_by
    * beginning_delta_gamma
    * detector_points
    * detector_edge_length
    * trajectory_points
    """

    beginning_x = 0
    beginning_y = 0
    beginning_bx = 0
    beginning_by = 0
    beginning_delta_gamma = 0

    def __init__(self, section_1, section_2, section_3):
        self.section_1 = section_1
        self.section_2 = section_2
        self.section_3 = section_3

    def section(self, i):
        if i == 1:
            return self.section_1
        elif i == 2:
            return self.section_2
        elif i == 3:
            return self.section_3
        else:
            assert False

    def setup(self):
        self.section_1.setup()
        self.section_2.setup()
        self.section_3.setup()

    @property
    def detector_points(self):
        return self._detector_points

    @detector_points.setter
    def detector_points(self, value):
        self._detector_points = value
        self.section_1.detector_points = value
        self.section_2.detector_points = value
        self.section_3.detector_points = value

    @property
    def detector_edge_length(self):
        return self._detector_edge_length

    @detector_edge_length.setter
    def detector_edge_length(self, value):
        self._detector_edge_length = value
        self.section_1.detector_edge_length = value
        self.section_2.detector_edge_length = value
        self.section_3.detector_edge_length = value

    def get_coords(self):
        coords_1 = (self.beginning_x, self.beginning_y, self.beginning_bx, self.beginning_by, self.beginning_delta_gamma)
        coords_2 = self.section_1.propagate_coords(coords_1)
        coords_3 = self.section_2.propagate_coords(coords_2)
        return coords_1, coords_2, coords_3

    def __transform_coordinates_section_1_to_floor(self, z, x, bz=None, bx=None):
        assert (bz is None and bx is None) or (bz is not None and bx is not None)
        z -= self.section_1.dipole_1_entrance_z
        x -= self.section_1.dipole_1_entrance_x_actual
        z, x = rotate(z, x, -self.section_1.reference_angle)
        if bz is None and bx is None:
            return z, x
        else:
            bz, bx = rotate(bz, bx, -self.section_1.reference_angle)
            return z, x, bz, bx

    def __dipole_2_entrance(self):
        return self.__transform_coordinates_section_1_to_floor(self.section_1.dipole_2_entrance_z, 0)

    def __transform_coordinates_section_2_to_floor(self, z, x, bz=None, bx=None):
        assert (bz is None and bx is None) or (bz is not None and bx is not None)
        # make coords relative to start of section 2 dipole 1
        z -= self.section_2.dipole_1_entrance_z
        x -= self.section_2.dipole_1_entrance_x_actual
        # add floor coordinates of section 1 dipole 2 entrance
        entrance_z, entrance_x = self.__dipole_2_entrance()
        z += entrance_z
        x += entrance_x
        if bz is None and bx is None:
            return z, x
        else:
            return z, x, bz, bx

    def __dipole_3_entrance(self):
        return self.__transform_coordinates_section_2_to_floor(self.section_2.dipole_2_entrance_z, 0)

    def __transform_coordinates_section_3_to_floor(self, z, x, bz=None, bx=None):
        assert (bz is None and bx is None) or (bz is not None and bx is not None)
        z -= self.section_3.dipole_1_entrance_z
        x -= self.section_3.dipole_1_entrance_x_actual
        z, x = rotate(z, x, -self.section_3.reference_angle)
        entrance_z, entrance_x = self.__dipole_3_entrance()
        z += entrance_z
        x += entrance_x
        if bz is None and bx is None:
            return z, x
        else:
            bz, bx = rotate(bz, bx, self.section_3.reference_angle)
            return z, x, bz, bx

    def __dipole_4_exit(self):
        return self.__transform_coordinates_section_3_to_floor(self.section_3.dipole_2_exit_z, 0)

    def __trajectory_to_floor_coordinates(self, trajectory, section):
        if section == 1:
            method = self.__transform_coordinates_section_1_to_floor
        elif section == 2:
            method = self.__transform_coordinates_section_2_to_floor
        elif section == 3:
            method = self.__transform_coordinates_section_3_to_floor
        else:
            assert False
        trajectory[2], trajectory[0], trajectory[5], trajectory[3] = method(trajectory[2], trajectory[0], trajectory[5], trajectory[3])
        return trajectory

    def track(self):
        """
        Tracks a particle through the chicane and returns a numpy array of shape
        (6, self.trajectory_points) where the first dimension corresponds to
        (x, y, z, bx, by, bz). These coordinates global floor coordinates.
        """
        coords_1, coords_2, coords_3 = self.get_coords()
        trajectory_1 = trajectory_to_array(self.section_1.track(coords_1))
        trajectory_2 = trajectory_to_array(self.section_2.track(coords_2))
        trajectory_3 = trajectory_to_array(self.section_3.track(coords_3))
        self.__trajectory_to_floor_coordinates(trajectory_1, 1)
        self.__trajectory_to_floor_coordinates(trajectory_2, 2)
        self.__trajectory_to_floor_coordinates(trajectory_3, 3)
        dipole_2_entrance_z, _ = self.__dipole_2_entrance()
        dipole_3_entrance_z, _ = self.__dipole_3_entrance()
        dipole_4_exit_z, _ = self.__dipole_4_exit()
        mask_1 = trajectory_1[2] < dipole_2_entrance_z
        mask_2 = np.logical_and(trajectory_2[2] > dipole_2_entrance_z, trajectory_2[2] < dipole_3_entrance_z)
        mask_3 = trajectory_3[2] > dipole_3_entrance_z
        trajectory_1 = trajectory_1[:, mask_1]
        trajectory_2 = trajectory_2[:, mask_2]
        trajectory_3 = trajectory_3[:, mask_3]
        trajectory = np.concatenate((trajectory_1, trajectory_2, trajectory_3), axis=1)
        assert np.all(np.diff(trajectory[2]) > 0)
        z_points = self.trajectory_points
        interpolated_trajectory = np.empty((6, z_points))
        interpolated_trajectory[2] = np.linspace(0, dipole_4_exit_z, z_points)
        for i in range(6):
            if i != 2:
                interpolated_trajectory[i] = scipy.interpolate.interp1d(trajectory[2], trajectory[i])(interpolated_trajectory[2])
        return interpolated_trajectory
