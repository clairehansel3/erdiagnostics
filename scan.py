from __future__ import print_function
print('<spinning up process>')
import sys
sys.path.append('/Users/claire/Desktop/software/SRW/env/work/srw_python')
sys.path.append('/afs/slac.stanford.edu/u/ra/chansel/edge-radiation/SRW/env/work/srw_python')
import erdiagnostics as er
import functools
import numpy as np
import os
import shutil

detector_points = int(sys.argv[2])
plot_points = 300
detector_edge_length = 0.04
ccd_edge_length = 0.005
trajectory_points = 10000
data_directory = '/scratch/chansel/data_{}'.format(detector_points)
data_directory_2 = '/Users/claire/Desktop/things/mount/scratch/chansel/data_{}'.format(detector_points)
plot_directory = '/Users/claire/Desktop/scan_results_{}'.format(detector_points)

class Scan(object):

    def __init__(self, name, set_scan_value, values, reference_value):
        self.name = name # name of the parameter being scanned
        self.set_scan_value = set_scan_value # function which takes a BC11 instance and a value and sets the parameter to that value
        self.values = values # list of values of the parameter to scan over
        self.reference_value = reference_value # the nominal value of the parameter
        assert reference_value in values

def try_mkdir(path):
    try:
        os.mkdir(path)
    except:
        pass

def initialize_mpi():
    sys.path.append('/afs/slac.stanford.edu/u/ra/chansel/edge-radiation/mpi4py-3.0.2/build/lib.linux-x86_64-2.7')
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print('process {} of {} initialized'.format(rank + 1, size))
    return rank, size, comm

def notify(message):
    pass #os.system('curl -s --form-string "token=acp78x6ov5dxvvpvajozzpkqidcr8a" --form-string "user=uqeggq7852j59vvxvu35s5unh8tbqd" --form-string message="{}" https://api.pushover.net/1/messages.json'.format(message))

def initialize_bc11():
    bc11 = er.BC11()
    bc11.detector_edge_length = detector_edge_length
    bc11.detector_points = detector_points
    bc11.trajectory_points = trajectory_points
    bc11.setup()
    return bc11

def to_readable_size(array_shape):
    """
    returns the number of bytes in a numpy array of shape 'array_shape' as a
    human readable string.
    """
    size = 8 * functools.reduce(lambda x, y: x * y, array_shape)
    if size > 1e10:
        raise Exception('Data file size would be larger than 10GB!')
    if size < 1e3:
        return '{:.2f}B'.format(size)
    elif size < 1e6:
        return '{:.2f}kB'.format(size / 1e3)
    elif size < 1e9:
        return '{:.2f}MB'.format(size / 1e6)
    else:
        return '{:.2f}GB'.format(size / 1e9)

def initialize_data(rank, size, comm, number_of_computations):
    d1_shape = (number_of_computations, 2, 9, plot_points, plot_points)
    d2_shape = (number_of_computations, 2, 3)
    d3_shape = (number_of_computations, 6, trajectory_points)
    if rank == 0:
        print('memory mapping data files:')
        print('intensity data  :', to_readable_size(d1_shape))
        print('overlap data    :', to_readable_size(d2_shape))
        print('trajectory data :', to_readable_size(d3_shape))
        intensity_data = np.memmap('{}/intensity'.format(data_directory), dtype=np.float64, mode='w+', shape=d1_shape)
        overlap_data = np.memmap('{}/overlap'.format(data_directory), dtype=np.float64, mode='w+', shape=d2_shape)
        trajectory_data = np.memmap('{}/trajectory'.format(data_directory), dtype=np.float64, mode='w+', shape=d3_shape)
        print('done mapping')
        for proc in range(1, size):
            print('root sending to process ', proc)
            comm.isend(None, dest=proc)
            print('root done')
        return intensity_data, overlap_data, trajectory_data
    else:
        print('process ', rank, 'blocking')
        comm.recv(source=0)
        print('process ', rank, 'recieved data mapped signal')
        intensity_data = np.memmap('{}/intensity'.format(data_directory), dtype=np.float64, mode='r+', shape=d1_shape)
        overlap_data = np.memmap('{}/overlap'.format(data_directory), dtype=np.float64, mode='r+', shape=d2_shape)
        trajectory_data = np.memmap('{}/trajectory'.format(data_directory), dtype=np.float64, mode='r+', shape=d3_shape)
        print('process ', rank, 'done mapping')
        return intensity_data, overlap_data, trajectory_data

def compute_scan_datum(computation_index, bc11, scan, value, intensity_data, overlap_data, trajectory_data):
    scan.set_scan_value(bc11, value)
    coords = bc11.get_coords()
    for i, section in enumerate((1, 2, 3)):
        for j, (dipole_1_on, dipole_2_on) in enumerate(((True, False), (False, True), (True, True))):
            wavefront_before_optics = bc11.section(section).compute_wavefront_at_detector(coords[i], dipole_1_on, dipole_2_on)
            wavefront_after_optics = bc11.section(section).propagate_wavefront_through_optics(wavefront_before_optics)
            intensity_data[computation_index, 0, 3 * i + j] = er.interpolate_intensity(er.wavefront_to_intensity(wavefront_before_optics), detector_edge_length, detector_edge_length, plot_points)
            intensity_data[computation_index, 1, 3 * i + j] = er.interpolate_intensity(er.wavefront_to_intensity(wavefront_after_optics), detector_edge_length, ccd_edge_length, plot_points)
        for k, edge_length in enumerate((detector_edge_length, ccd_edge_length)):
            for l, (m, n) in enumerate(((0, 1), (3, 4), (6, 7))):
                overlap_data[computation_index, k, l] = er.compute_overlap_intensity(intensity_data[computation_index, k, m], intensity_data[computation_index, k, n], edge_length)
    trajectory_data[i, :, :] = bc11.track()
    scan.set_scan_value(bc11, scan.reference_value)

def compute_scans(scans):
    try_mkdir(data_directory)
    rank, size, comm = initialize_mpi()
    bc11 = initialize_bc11()
    computations = []
    process = 0
    for scan in scans:
        for value in scan.values:
            computations.append([scan, value, process])
            process += 1
            if process == size:
                process = 0
    data = initialize_data(rank, size, comm, len(computations))
    for i, (scan, value, process) in enumerate(computations):
        if rank == process:
            print('process {} computing scan {} value {}'.format(rank, scan.name, value))
            compute_scan_datum(i, bc11, scan, value, *data)
    if rank == 0:
        print('process 0 waiting for done signals')
        for i in range(1, size):
            print('process 0 blocking for process {}'.format(i))
            comm.recv(source=i, tag=12)
            print('process 0 recieved done signal from process {}'.format(i))
        print('process 0 terminating')
        notify('computation done')
    else:
        print('process {} sending done signal'.format(rank))
        comm.send(None, dest=0, tag=12)
        print('process {} terminating'.format(rank))

def create_movie(scan, intensity, before_or_after_optics, detector_edge_length_or_ccd_edge_length):
    x_min = -0.5 * detector_edge_length_or_ccd_edge_length
    x_max = 0.5 * detector_edge_length_or_ccd_edge_length
    x_num = detector_points
    y_min = -0.5 * detector_edge_length_or_ccd_edge_length
    y_max = 0.5 * detector_edge_length_or_ccd_edge_length
    y_num = detector_points
    x_step = (x_max - x_min) / (x_num - 1)
    y_step = (y_max - y_min) / (y_num - 1)
    x = np.linspace(x_min - 0.5 * x_step, x_max + 0.5 * x_step, x_num + 1)
    y = np.linspace(y_min - 0.5 * y_step, y_max + 0.5 * y_step, y_num + 1)
    i1_max = np.max(intensity[:, 2, :, :])
    i2_max = np.max(intensity[:, 5, :, :])
    i3_max = np.max(intensity[:, 8, :, :])
    for w in range(len(scan.values)):
        fig = plt.figure(figsize=(10, 3))
        plt.subplot(131)
        plt.title('Detector 1 (' + before_or_after_optier.replace('_', ' ').title() + ')', fontsize=8)
        plt.pcolormesh(1000 * x, 1000 * y, intensity[w, 2, :, :], vmin=0, vmax=i1_max, cmap='inferno')
        plt.gca().set_aspect(1)
        plt.colorbar(fraction=0.046, pad=0.04).set_label('ph/s/.1%bw/mm^2', fontsize=8)
        plt.xlabel('x (mm)', fontsize=8)
        plt.ylabel('y (mm)', fontsize=8)
        plt.subplot(132)
        plt.title('Detector 2 (' + before_or_after_optier.replace('_', ' ').title() + ')', fontsize=8)
        plt.pcolormesh(1000 * x, 1000 * y, intensity[w, 5, :, :], vmin=0, vmax=i2_max, cmap='inferno')
        plt.gca().set_aspect(1)
        plt.colorbar(fraction=0.046, pad=0.04).set_label('ph/s/.1%bw/mm^2', fontsize=8)
        plt.xlabel('x (mm)', fontsize=8)
        plt.ylabel('y (mm)', fontsize=8)
        plt.subplot(133)
        plt.title('Detector 3 (' + before_or_after_optier.replace('_', ' ').title() + ')', fontsize=8)
        plt.pcolormesh(1000 * x, 1000 * y, intensity[w, 8, :, :], vmin=0, vmax=i3_max, cmap='inferno')
        plt.gca().set_aspect(1)
        plt.colorbar(fraction=0.046, pad=0.04).set_label('ph/s/.1%bw/mm^2', fontsize=8)
        plt.xlabel('x (mm)', fontsize=8)
        plt.ylabel('y (mm)', fontsize=8)
        fig.suptitle(scan.name.replace('_', ' ').title() + ' = ' + '{} mm'.format(int(round(1000 * scan.values[w]))))
        plt.tight_layout()
        plt.savefig('{}/{}/frames/{}_{}.png'.format(plot_directory, scan.name, before_or_after_optics, w), dpi=400)
        plt.close(fig)
    if os.path.isfile('{}/{}/movie_{}.mp4'.format(plot_directory, scan.name, before_or_after_optics)):
        os.remove('{}/{}/movie_{}.mp4'.format(plot_directory, scan.name, before_or_after_optics))
    os.system('ffmpeg -framerate 10 -i \'{}/{}/frames/{}_%d.png\' -c:v libx264 -profile:v high -crf 30 -pix_fmt yuv420p {}/{}/movie_{}.mp4'.format(plot_directory, scan.name, before_or_after_optics, plot_directory, scan.name, before_or_after_optics))

def analyze_scan(scan, intensities, overlaps, trajectories, intensity_names, intensity_filenames):
    create_movie(scan, intensities[:, 0, :, :, :], 'before_optics', detector_edge_length)
    create_movie(scan, intensities[:, 1, :, :, :], 'after_optics', ccd_edge_length)
    parameter_value_labels = [scan.name + '=' + str(value) for value in scan.values]
    reference_data_index = np.where(np.array(scan.values) == float(scan.reference_value))[0][0]
    remove_reference_data = lambda arr: [arr[i] for i in range(len(arr)) if i != reference_data_index]
    er.plot_relative_trajectories(trajectories[reference_data_index], remove_reference_data(parameter_value_labels), remove_reference_data(trajectories), filename='{}/{}/rel_traj.png'.format(plot_directory, scan.name))
    for j, optics_on_or_off in enumerate(('optics_off', 'optics_on')):
        relative_overlaps = overlaps[:, j, :] / overlaps[reference_data_index, j, :]
        for i in range(3):
            plt.title('Detector {} Overlap ('.format(i + 1) + optics_on_or_off.replace('_', ' ').title() + ')')
            plt.xlabel(scan.name.replace('_', ' ').title() + ' (mm)')
            plt.ylabel('Normalized Overlap')
            plt.plot(scan.values * 1000, relative_overlaps[:, i])
            plt.tight_layout()
            plt.savefig('{}/{}/{}_rel_overlap_{}.png'.format(plot_directory, scan.name, optics_on_or_off, i))
            plt.clf()
        plt.title('All Detectors Overlap (' + optics_on_or_off.replace('_', ' ').title() + ')')
        plt.plot(scan.values * 1000, relative_overlaps[:, 0], label='detector 1')
        plt.plot(scan.values * 1000, relative_overlaps[:, 1], label='detector 2')
        plt.plot(scan.values * 1000, relative_overlaps[:, 2], label='detector 3')
        plt.xlabel(scan.name.replace('_', ' ').title() + ' (mm)')
        plt.ylabel('Normalized Overlap')
        plt.legend()
        plt.tight_layout()
        plt.savefig('{}/{}/{}_rel_overlap.png'.format(plot_directory, scan.name, optics_on_or_off))
        plt.clf()
    for j, optics_on_or_off in enumerate(('optics_off', 'optics_on')):
        for i in range(len(scan.values)):
            for k in range(9):
                title = intensity_names[k] + ' (' + optics_on_or_off.replace('_', ' ').title() + '), ' + parameter_value_labels[i]
                fname = '{}/{}/intensity/'.format(plot_directory, scan.name) + intensity_filenames[k] + '_{}_{}.png'.format(optics_on_or_off, i)
                er.plot_intensity(intensities[i, j, k], detector_edge_length if optics_on_or_off == 'optics_off' else ccd_edge_length, plot_title=title, filename=fname)

def analyze_scans(scans):
    try_mkdir(plot_directory)
    try_mkdir('{}/data'.format(plot_directory))
    if not all(os.path.isfile('{}/data/{}'.format(plot_directory, file_name)) for file_name in ('intensity', 'overlap', 'trajectory')):
        for file_name in ('intensity', 'overlap', 'trajectory'):
            shutil.copy('{}/{}'.format(data_directory_2, file_name), '{}/data/{}'.format(plot_directory, file_name))
    number_of_computations = 0
    for scan in scans:
        for value in scan.values:
            number_of_computations += 1
    d1_shape = (number_of_computations, 2, 9, plot_points, plot_points)
    d2_shape = (number_of_computations, 2, 3)
    d3_shape = (number_of_computations, 6, trajectory_points)
    intensity_data = np.memmap('{}/data/intensity'.format(plot_directory), dtype=np.float64, mode='r+', shape=d1_shape)
    overlap_data = np.memmap('{}/data/overlap'.format(plot_directory), dtype=np.float64, mode='r+', shape=d2_shape)
    trajectory_data = np.memmap('{}/data/trajectory'.format(plot_directory), dtype=np.float64, mode='r+', shape=d3_shape)
    intensity_names = []
    intensity_filenames = []
    for section in (1, 2, 3):
        for dipole in (1, 2):
            intensity_names.append('Section {} Dipole {} Only'.format(section, dipole))
            intensity_filenames.append('section_{}_dipole_{}'.format(section, dipole))
        intensity_names.append('Section {} Dipoles 1 & 2'.format(section))
        intensity_filenames.append('section_{}_dipoles_12'.format(section))
    computation_index = 0
    for scan in scans:
        try_mkdir('{}/{}'.format(plot_directory, scan.name))
        try_mkdir('{}/{}/intensity'.format(plot_directory, scan.name))
        try_mkdir('{}/{}/frames'.format(plot_directory, scan.name))
        analyze_scan(
            scan,
            intensity_data[computation_index:computation_index + len(scan.values)],
            overlap_data[computation_index:computation_index + len(scan.values)],
            trajectory_data[computation_index:computation_index + len(scan.values)],
            intensity_names, intensity_filenames
        )
        computation_index += len(scan.values)
    notify('analysis done')

def run(scans):
    if sys.argv[1] == 'run':
        compute_scans(scans)
    elif sys.argv[1] == 'analyze':
        global plt
        import matplotlib.pyplot as plt
        analyze_scans(scans)

scans = [
    Scan(
        'beginning_x',
        lambda self, value: (
            setattr(self, 'beginning_x', value)
        ),
        np.linspace(-0.01, 0.01, 21),
        0.0
    ),
    Scan(
        'beginning_y',
        lambda self, value: (
            setattr(self, 'beginning_y', value)
        ),
        np.linspace(-0.01, 0.01, 21),
        0.0
    ),
    Scan(
        'beginning_bx',
        lambda self, value: (
            setattr(self, 'beginning_bx', value)
        ),
        np.linspace(-1 * np.pi / 180, 1 * np.pi / 180, 21),
        0.0
    ),
    Scan(
        'beginning_by',
        lambda self, value: (
            setattr(self, 'beginning_by', value)
        ),
        np.linspace(-1 * np.pi / 180, 1 * np.pi / 180, 21),
        0.0
    ),
    Scan(
        'beginning_delta_gamma',
        lambda self, value: (
            setattr(self, 'beginning_delta_gamma', value)
        ),
        np.linspace(-10 * np.pi / 180, 10 * np.pi / 180, 21),
        0.0
    ),
    Scan(
        'q1_dx',
        lambda self, value: (
            setattr(self.section_1, 'quadrupole_offset_x', value),
            self.setup()
        ),
        np.linspace(-0.01, 0.01, 21),
        0.0
    ),
    Scan(
        'q1_dy',
        lambda self, value: (
            setattr(self.section_1, 'quadrupole_offset_y', value),
            self.setup()
        ),
        np.linspace(-0.01, 0.01, 21),
        0.0
    ),
    Scan(
        'q2_dx',
        lambda self, value: (
            setattr(self.section_3, 'quadrupole_1_offset_x', value),
            self.setup()
        ),
        np.linspace(-0.01, 0.01, 21),
        0.0
    ),
    Scan(
        'q2_dy',
        lambda self, value: (
            setattr(self.section_3, 'quadrupole_1_offset_y', value),
            self.setup()
        ),
        np.linspace(-0.01, 0.01, 21),
        0.0
    ),
    Scan(
        'q3_dx',
        lambda self, value: (
            setattr(self.section_3, 'quadrupole_2_offset_x', value),
            self.setup()
        ),
        np.linspace(-0.01, 0.01, 21),
        0.0
    ),
    Scan(
        'q3_dy',
        lambda self, value: (
            setattr(self.section_3, 'quadrupole_2_offset_y', value),
            self.setup()
        ),
        np.linspace(-0.01, 0.01, 21),
        0.0
    )
]

if __name__ == '__main__':
    run(scans)
