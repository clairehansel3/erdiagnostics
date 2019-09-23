from __future__ import print_function
import sys
sys.path.append('/Users/claire/Desktop/software/SRW/env/work/srw_python')
sys.path.append('/afs/slac.stanford.edu/u/ra/chansel/edge-radiation/SRW/env/work/srw_python')
import erdiagnostics as er
import os
import pickle

bc11 = er.BC11()
bc11.detector_edge_length = float(sys.argv[2])
bc11.detector_points = 256
bc11.trajectory_points = 10000
bc11.setup()

ccd_edge_length = 0.005
plot_points = 1000
data_file_name = 'nominal_data_{}'.format(bc11.detector_edge_length)

if sys.argv[1] == 'run':
    coords_1, coords_2, coords_3 = bc11.get_coords()
    wavefront_1 = bc11.section_1.compute_wavefront_at_detector(coords_1, True, True)
    wavefront_2 = bc11.section_2.compute_wavefront_at_detector(coords_2, True, True)
    wavefront_3 = bc11.section_3.compute_wavefront_at_detector(coords_3, True, True)
    with open(data_file_name, 'wb') as f:
        pickle.dump([wavefront_1, wavefront_2, wavefront_3], f)
elif sys.argv[1] == 'analyze':
    with open(data_file_name, 'rb') as f:
        wavefronts = pickle.load(f)
    for section in (1, 2, 3):
        before_optics_wavefront = wavefronts[section - 1]
        after_optics_wavefront = bc11.section(section).propagate_wavefront_through_optics(before_optics_wavefront)
        er.plot_intensity(
            er.wavefront_to_intensity(before_optics_wavefront),
            bc11.detector_edge_length,
            plot_title='Detector {}, Before Optics'.format(section),
            filename='detector_{}_before_optics_{}.png'.format(section, bc11.detector_edge_length),
        )
        er.plot_intensity(
            er.interpolate_intensity(
                er.wavefront_to_intensity(after_optics_wavefront),
                bc11.detector_edge_length,
                ccd_edge_length,
                intensity_points_new = plot_points
            ),
            ccd_edge_length,
            plot_title='Detector {}, After Optics'.format(section),
            filename='detector_{}_after_optics_{}.png'.format(section, bc11.detector_edge_length),
        )
