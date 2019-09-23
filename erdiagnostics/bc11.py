# -*- coding: utf-8 -*-
from .chicane import Chicane
from .dogleg import Dogleg
from .setup_object import ensure_setup
import copy
import srwlib

class BC11Section(Dogleg):

    attributes_requiring_resetup = Dogleg.attributes_requiring_resetup + [
        'focal_length',
    ]

    reference_energy_mev = 335
    reference_particle_alignment_cycles = 5
    dipole_1_pad_drift_length = 0.3
    dipole_1_strength = 0.51673286
    dipole_1_arc_length = 0.2035002175
    dipole_1_edge_length = 0.07
    # dipole_1_polarity has no default value
    # dipole_separation has no default value
    dipole_2_strength = 0.51673286
    dipole_2_arc_length = 0.2035002175
    dipole_2_edge_length = 0.07
    # dipole_2_polarity has no default value
    dipole_2_pad_drift_length = 0.3
    detector_distance = 1
    detector_edge_length = (0.01 * 2.54) * 2 # this is the aperature diameter too
    # detector_points has no default value
    detector_wavelength = 600
    average_current = 0.5
    relative_precision = 1e-5
    trajectory_points = 100000
    focal_length = 1

    def setup(self):
        super(BC11Section, self).setup()
        aperature = srwlib.SRWLOptA('c', 'a', self.detector_edge_length, 0, 0, 0)
        aperature_params = [0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0]
        lens = srwlib.SRWLOptL(self.focal_length, self.focal_length)
        lens_params = [0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0]
        drift = srwlib.SRWLOptD(self.focal_length)
        drift_params = [0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0]
        self.__optics = srwlib.SRWLOptC([aperature, lens, drift], [aperature_params, lens_params, drift_params])

    @ensure_setup
    def propagate_wavefront_through_optics(self, wavefront):
        wavefront_copy = copy.deepcopy(wavefront)
        srwlib.srwl.PropagElecField(wavefront_copy, self.__optics)
        return wavefront_copy

class BC11Section1(BC11Section):

    attributes_requiring_resetup = BC11Section.attributes_requiring_resetup + [
        'drift_1_length',
        'quadrupole_strength',
        'quadrupole_offset_x',
        'quadrupole_offset_y',
        'quadrupole_length',
        'quadrupole_edge_length',
        'drift_2_length'
    ]

    drift_1_length = 0.24586885
    quadrupole_strength = 1.9444444444
    quadrupole_offset_x = 0
    quadrupole_offset_y = 0
    quadrupole_length = 0.108
    quadrupole_edge_length = 0.07
    drift_2_length = 2.09185209
    dipole_1_polarity = -1
    dipole_2_polarity = 1

    def get_dipole_separation(self):
        return self.drift_1_length + self.quadrupole_length + self.drift_2_length

    def add_beam_optics(self, magnetic_field_container):
        quadrupole_1 = srwlib.SRWLMagFldM(self.quadrupole_strength, 2, 'n', self.quadrupole_length, self.quadrupole_edge_length, 0)
        magnetic_field_container.add(quadrupole_1, self.quadrupole_offset_x, self.quadrupole_offset_y, self.dipole_1_exit_z + self.drift_1_length + 0.5 * self.quadrupole_length)

class BC11Section2(BC11Section):

    dipole_separation = 0.8302
    dipole_1_polarity = 1
    dipole_2_polarity = 1

class BC11Section3(BC11Section):

    attributes_requiring_resetup = BC11Section.attributes_requiring_resetup + [
        'drift_1_length',
        'quadrupole_1_strength',
        'quadrupole_1_offset_x',
        'quadrupole_1_offset_y',
        'quadrupole_1_length',
        'quadrupole_1_edge_length',
        'drift_2_length',
        'quadrupole_2_strength',
        'quadrupole_2_offset_x',
        'quadrupole_2_offset_y',
        'quadrupole_2_length',
        'quadrupole_2_edge_length',
        'drift_3_length'
    ]

    drift_1_length = 0.28846226498102677
    quadrupole_1_strength = 1.9444444444
    quadrupole_1_offset_x = 0
    quadrupole_1_offset_y = 0
    quadrupole_1_length = 0.108
    quadrupole_1_edge_length = 0.07
    drift_2_length = 1.695389782
    quadrupole_2_strength = 1.9444444444
    quadrupole_2_offset_x = 0
    quadrupole_2_offset_y = 0
    quadrupole_2_length = 0.108
    quadrupole_2_edge_length = 0.07
    drift_3_length = 0.245868846721
    dipole_1_polarity = 1
    dipole_2_polarity = -1

    def get_dipole_separation(self):
        return self.drift_1_length + self.quadrupole_1_length + self.drift_2_length + self.quadrupole_2_length + self.drift_3_length

    def add_beam_optics(self, magnetic_field_container):
        quadrupole_1 = srwlib.SRWLMagFldM(self.quadrupole_1_strength, 2, 's', self.quadrupole_1_length, self.quadrupole_1_edge_length, 0)
        magnetic_field_container.add(quadrupole_1, self.quadrupole_1_offset_x, self.quadrupole_1_offset_y, self.dipole_1_exit_z + self.drift_1_length + 0.5 * self.quadrupole_1_length)
        quadrupole_2 = srwlib.SRWLMagFldM(self.quadrupole_2_strength, 2, 'n', self.quadrupole_2_length, self.quadrupole_2_edge_length, 0)
        magnetic_field_container.add(quadrupole_2, self.quadrupole_2_offset_x, self.quadrupole_2_offset_y, self.dipole_1_exit_z + self.drift_1_length + self.quadrupole_1_length + self.drift_2_length + 0.5 * self.quadrupole_2_length)

class BC11(Chicane):

    def __init__(self):
        super(BC11, self).__init__(
            BC11Section1(),
            BC11Section2(),
            BC11Section3()
        )
