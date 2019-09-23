# -*- coding: utf-8 -*-
from .misc import track_drift, rotate
from .setup_object import SetupObject, ensure_setup
import numpy as np
import scipy.interpolate
import srwlib

class Dogleg(SetupObject):
    """
    A dogleg consists of two bending magnets and a detector. By default there
    are no beam optics in between the bending magnets but a dogleg with beam
    optics can be modeled by subclassing this class and override
    '.add_beam_optics()'. You may also have to override
    '.get_dipole_separation()' if the beam optics length depends on parameters
    of your subclass (see docstring for '.get_dipole_separation()').

    Attributes Requiring a Call to '.setup()' if Changed:
    * reference_energy_mev
    * reference_particle_alignment_cycles
    * dipole_1_strength
    * dipole_1_arc_length
    * dipole_1_edge_length
    * dipole_1_polarity
    * dipole_1_pad_drift_length
    * dipole_separation (note: see docstring or '.get_dipole_separation()')
    * dipole_2_strength
    * dipole_2_arc_length
    * dipole_2_edge_length
    * dipole_2_polarity
    * dipole_2_pad_drift_length

    Attributes Computed During Setup (Dependent on the Above Attributes)
    * dipole_separation (note: see docstring or '.get_dipole_separation()')
    * reference_gamma
    * reference_beta
    * dipole_1_rho
    * dipole_1_angle
    * dipole_1_length_z
    * dipole_1_entrance_z
    * dipole_1_exit_z
    * dipole_2_rho
    * dipole_2_angle
    * dipole_2_length_z
    * dipole_2_entrance_z
    * dipole_2_exit_z
    * detector_z
    * z_at_end
    * reference_angle
    * reference_offset
    * reference_trajectory

    Other Attributes:
    * relative_precision
    * trajectory_points
    * detector_edge_length
    * detector_points
    * detector_distance
    * detector_wavelength
    * average_current
    """

    attributes_requiring_resetup = [
        'reference_energy_mev',
        'reference_particle_alignment_cycles',
        'dipole_1_strength',
        'dipole_1_arc_length',
        'dipole_1_edge_length',
        'dipole_1_polarity',
        'dipole_1_pad_drift_length',
        'dipole_separation',
        'dipole_2_strength',
        'dipole_2_arc_length',
        'dipole_2_edge_length',
        'dipole_2_polarity',
        'dipole_2_pad_drift_length',
    ]

    def setup(self):
        """
        Defines various helpful attributes, builds the fields, and aligns the
        reference trajectory.
        """
        self.dipole_separation = self.get_dipole_separation()
        super(Dogleg, self).setup()
        # define helpful attributes
        self.reference_gamma = self.reference_energy_mev / 0.5109989461
        self.reference_beta = np.sqrt(1 - self.reference_gamma ** -2)
        self.dipole_1_rho = self.reference_gamma * self.reference_beta * 299792458 / (1.758820024e11 * self.dipole_1_strength)
        self.dipole_1_angle = self.dipole_1_arc_length / self.dipole_1_rho
        self.dipole_1_length_z = self.dipole_1_rho * np.sin(self.dipole_1_angle)
        self.dipole_1_entrance_z = self.dipole_1_pad_drift_length * np.cos(self.dipole_1_angle)
        self.dipole_1_exit_z = self.dipole_1_entrance_z + self.dipole_1_length_z
        self.dipole_2_rho = self.reference_gamma * self.reference_beta * 299792458 / (1.758820024e11 * self.dipole_2_strength)
        self.dipole_2_angle = self.dipole_2_arc_length / self.dipole_2_rho
        self.dipole_2_length_z = self.dipole_2_rho * np.sin(self.dipole_2_angle)
        self.dipole_2_entrance_z = self.dipole_1_exit_z + self.dipole_separation
        self.dipole_2_exit_z = self.dipole_2_entrance_z + self.dipole_1_length_z
        self.detector_z = self.dipole_2_entrance_z + self.detector_distance
        self.z_at_end = self.dipole_2_exit_z + self.dipole_2_pad_drift_length * np.cos(self.dipole_2_angle)
        # build fields
        self.__magnetic_field_regular = srwlib.SRWLMagFldC()
        self.__magnetic_field_no_dipole_1 = srwlib.SRWLMagFldC()
        self.__magnetic_field_no_dipole_2 = srwlib.SRWLMagFldC()
        self.__magnetic_field_only_dipole_1 = srwlib.SRWLMagFldC()
        self.__magnetic_field_regular.allocate(0)
        self.__magnetic_field_no_dipole_1.allocate(0)
        self.__magnetic_field_no_dipole_2.allocate(0)
        self.__magnetic_field_only_dipole_1.allocate(0)
        # dipole 1
        dipole_1 = srwlib.SRWLMagFldM(self.dipole_1_polarity * self.dipole_1_strength, 1, 'n', self.dipole_1_length_z, self.dipole_1_edge_length, 0)
        self.__magnetic_field_regular.add(dipole_1, 0, 0, self.dipole_1_entrance_z + 0.5 * self.dipole_1_length_z)
        self.__magnetic_field_no_dipole_2.add(dipole_1, 0, 0, self.dipole_1_entrance_z + 0.5 * self.dipole_1_length_z)
        self.__magnetic_field_only_dipole_1.add(dipole_1, 0, 0, self.dipole_1_entrance_z + 0.5 * self.dipole_1_length_z)
        # beam optics between dipoles
        self.add_beam_optics(self.__magnetic_field_regular)
        self.add_beam_optics(self.__magnetic_field_no_dipole_1)
        self.add_beam_optics(self.__magnetic_field_no_dipole_2)
        # dipole 2
        dipole_2 = srwlib.SRWLMagFldM(self.dipole_2_polarity * self.dipole_2_strength, 1, 'n', self.dipole_2_length_z, self.dipole_2_edge_length, 0)
        self.__magnetic_field_regular.add(dipole_2, 0, 0, self.dipole_2_entrance_z + 0.5 * self.dipole_2_length_z)
        self.__magnetic_field_no_dipole_1.add(dipole_2, 0, 0, self.dipole_2_entrance_z + 0.5 * self.dipole_2_length_z)
        # align reference trajectory
        self.align_reference_trajectory()

    def get_dipole_separation(self):
        """
        If you are creating a Dogleg subclass with, for example, a quadrupole
        and a drift between the dipoles, you would want your subclass to have
        the attributes 'quadrupole_length' and 'drift_length'. In this case,
        'dipole_separation' is no longer an independent variable but rather it
        should be equal to the sum of 'quadrupole_length' and 'drift_length'.
        What you should do in this case is override this method so it returns
        the sum of 'quadrupole_length' and 'drift_length'. However if the Dogleg
        subclass has no beam optics between the dipoles, 'dipole_separation' is
        just a regular independent variable and thus this method should not be
        overridden.
        """
        return self.dipole_separation

    def add_beam_optics(self, magnetic_field_container):
        """
        This method should be overridden in order to add beam optics between
        dipoles 1 and 2. Remember that if you add optics, they should be
        contained between 'self.dipole_1_exit_z' and 'self.dipole_2_entrance_z'.
        'magnetic_field_container' is a srwlib.SRWLMagFldC instance.
        """
        pass

    def align_reference_trajectory(self):
        """
        This function adjusts the initial conditions of the reference particle
        so that after the first dipole, the reference trajectory is aligned to
        x = y = 0. This is done iteratively. Note that the reference particle
        starts with y = y' = z = 0
        """
        # the initial offset of the particle from the axis in the x direction
        self.reference_offset = 0
        # angle the reference particle's initial velocity makes with the z axis
        self.reference_angle = 0
        # iteratively align the trajectory
        for i in range(self.reference_particle_alignment_cycles):
            trajectory = self.track((0, 0, 0, 0, 0), 'only_dipole_1')
            self.reference_offset -= trajectory.arX[-1]
            self.reference_angle -= np.arctan2(trajectory.arXp[-1], trajectory.arZp[-1])
        # set the reference trajectory
        self.reference_trajectory = self.track((0, 0, 0, 0, 0), 'regular')
        # compute the x position at which the reference particle enters dipole 1
        # note that this is the 'actual' entrance x which accounts for fringe
        # fields from dipole 1 as opposed to the 'ideal' entrance x used in
        # '.transform_coordinates()'
        self.dipole_1_entrance_x_actual = scipy.interpolate.interp1d(
            np.array(self.reference_trajectory.arZ),
            np.array(self.reference_trajectory.arX),
            kind='linear'
        )(self.dipole_1_entrance_z)

    @ensure_setup
    def transform_coordinates(self, coords, dipole_1_on=True):
        """
        This function takes the coordinates of a particle at the entrance of the
        first dipole relative to the reference trajectory and converts them into
        the coordinate system used internally by the dogleg.
        """
        x, y, bx, by, delta_gamma = coords
        gamma = self.reference_gamma + delta_gamma
        bz = np.sqrt(1 - gamma ** -2)
        if dipole_1_on:
            # rotate the coordinates by the reference angle
            z, x = rotate(0, x, self.reference_angle)
            bz, bx = rotate(bz, bx, self.reference_angle)
            # get the coordinates to the 'ideal' entrance of dipole 1 (i.e. where
            # the reference particle would enter dipole 1 if all fringe effects were
            # turned off)
            ideal_entrance_z = self.dipole_1_entrance_z
            ideal_entrance_x = self.reference_offset - self.dipole_1_pad_drift_length * np.sin(self.reference_angle)
            # add the coordinates of the entrance of the drift to the particle
            z += ideal_entrance_z
            x += ideal_entrance_x
        else:
            # no rotation needed
            z = self.dipole_1_entrance_z
        # analytically track the particle backwards to z = 0 while ignoring
        # fringe effects of dipole 1
        ct = -z / bz
        x += ct * bx
        y += ct * by
        # return the new coordinates
        return x, y, 0.0, bx, by, gamma

    @ensure_setup
    def track(self, coords, field_type='regular'):
        """
        Takes the coordinates of a particle relative to the reference trajectory
        and tracks it through the dogleg, returning a srwlib.SRWLPrtTrj
        instance. The returned trajectory is in the local internal coordinate
        system. 'field_type' specifies which magnetic field to track through,
        where different options turn off magnets by replacing them with a drift
        of the same length.
        """
        fields = {
            'regular': self.__magnetic_field_regular,
            'no_dipole_1': self.__magnetic_field_no_dipole_1,
            'no_dipole_2': self.__magnetic_field_no_dipole_2,
            'only_dipole_1': self.__magnetic_field_only_dipole_1
        }
        assert field_type in fields.keys()
        magnetic_field = fields[field_type]
        particle = srwlib.SRWLParticle(*self.transform_coordinates(coords, field_type != 'no_dipole_1'))
        trajectory = srwlib.SRWLPrtTrj()
        trajectory.partInitCond = particle
        trajectory.allocate(self.trajectory_points)
        trajectory.ctStart = 0
        trajectory.ctEnd = self.z_at_end
        if len(magnetic_field.arMagFld) == 0:
            return track_drift(trajectory)
        return srwlib.srwl.CalcPartTraj(trajectory, magnetic_field, [1])

    @ensure_setup
    def propagate_coords(self, coords):
        """
        Takes the coordinates of a particle relative to the reference trajectory
        and tracks it through the dogleg. Returns the relative coordinates of
        the particle right before entering dipole 2. Dipole 2 is turned off when
        tracking so fringe field effects from it are ignored.
        """
        # track
        trajectory = self.track(coords, 'no_dipole_2')
        # get coordinates at z = self.z_at_end
        x, bx, y, by, z, bz = (getattr(trajectory, name)[-1] for name in ('arX', 'arXp', 'arY', 'arYp', 'arZ', 'arZp'))
        # analytically track the particle backward through a drift to z =
        # self.dipole_2_entrance_z
        ct = (self.dipole_2_entrance_z - z) / bz
        x += ct * bx
        y += ct * by
        # return coordinates
        return x, y, bx, by, coords[-1]

    @ensure_setup
    def compute_wavefront_at_detector(self, coords, dipole_1_on=True, dipole_2_on=True):
        """
        computes the wavefront on the detector's surface where the wavelength is
        given by 'self.detector_wavelength'
        """
        # get magnetic field
        if dipole_1_on and dipole_2_on:
            magnetic_field = self.__magnetic_field_regular
        elif not dipole_1_on and dipole_2_on:
            magnetic_field = self.__magnetic_field_no_dipole_1
        elif dipole_1_on and not dipole_2_on:
            magnetic_field = self.__magnetic_field_no_dipole_2
        else:
            raise Exception('At least one dipole must be on to compute wavefront at detector')
        # ensure paricles won't be tracked past the detector
        if self.detector_z <= self.z_at_end:
            raise Exception('Particles tracked past detector, please decrease dipole 2 pad drift length or increase detector distance')
        # define initial beam
        beam = srwlib.SRWLPartBeam()
        beam.Iavg = self.average_current
        beam.partStatMom1 = srwlib.SRWLParticle(*self.transform_coordinates(coords, dipole_1_on))
        # define wavefront
        wavefront = srwlib.SRWLWfr()
        wavefront.allocate(1, self.detector_points, self.detector_points)
        wavefront.mesh.eStart = 1239.8 / self.detector_wavelength
        wavefront.mesh.eFin = 1239.8 / self.detector_wavelength
        wavefront.mesh.xStart = -0.5 * self.detector_edge_length
        wavefront.mesh.xFin = 0.5 * self.detector_edge_length
        wavefront.mesh.yStart = -0.5 * self.detector_edge_length
        wavefront.mesh.yFin = 0.5 * self.detector_edge_length
        wavefront.mesh.zStart = self.detector_z
        wavefront.partBeam = beam
        # compute wavefront
        precision = [0, self.relative_precision, 0, self.z_at_end, self.trajectory_points, 1, -1]
        srwlib.srwl.CalcElecFieldSR(wavefront, 0, magnetic_field, precision)
        return wavefront
