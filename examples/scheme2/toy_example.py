import os
import time
import shutil
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.comparison as sc
import mountainsort5 as ms5
print(ms5.__file__)
from generate_visualization_output import generate_visualization_output
from spikeforest.load_spikeforest_recordings.SFRecording import SFRecording
import spikeinterface as si
import numpy as np
import matplotlib.pyplot as plt

def main(num_cores):
    recording, sorting_true = se.toy_example(duration=60 * 30, num_channels=16, num_units=32, sampling_frequency=30000, num_segments=1, seed=0)

    timer = time.time()

    # lazy preprocessing
    recording_filtered = spre.bandpass_filter(recording, freq_min=300, freq_max=6000)
    recording_preprocessed: si.BaseRecording = spre.whiten(recording_filtered, dtype='float32')

    # sorting
    print('Starting MountainSort5 (scheme 2)')
    sorting = ms5.sorting_scheme2(
        recording_preprocessed,
        sorting_parameters=ms5.Scheme2SortingParameters(
            phase1_detect_channel_radius=150,
            detect_channel_radius=50,
            training_duration_sec=60
        ),
        num_cores=num_cores
    )
    
    elapsed_sec = time.time() - timer
    duration_sec = recording.get_total_duration()
    print(f'Elapsed time for sorting: {elapsed_sec:.2f} sec -- x{(duration_sec / elapsed_sec):.2f} speed compared with real time for {recording.get_num_channels()} channels')

    print('Comparing with truth')
    comparison = sc.compare_sorter_to_ground_truth(gt_sorting=sorting_true, tested_sorting=sorting)
    print(comparison.get_performance())

    #######################################################################

    if os.getenv('GENERATE_VISUALIZATION_OUTPUT') == '1':
        if os.path.exists('output/toy_example'):
            shutil.rmtree('output/toy_example')
        rec = SFRecording({
            'name': 'toy_example',
            'studyName': 'toy_example',
            'studySetName': 'toy_example',
            'sampleRateHz': recording_preprocessed.get_sampling_frequency(),
            'numChannels': recording_preprocessed.get_num_channels(),
            'durationSec': recording_preprocessed.get_total_duration(),
            'numTrueUnits': sorting_true.get_num_units(),
            'sortingTrueObject': {},
            'recordingObject': {}
        })
        generate_visualization_output(rec=rec, recording_preprocessed=recording_preprocessed, sorting=sorting, sorting_true=sorting_true)

if __name__ == '__main__':
    num_trials = 5
    max_cores = os.cpu_count()
    core_values = [1] + list(range(2, max_cores - 1, 2))
    all_times = []

    for num_cores in core_values:
        times_for_this_core = []
        for trial in range(num_trials):
            start_time = time.time()
            main(num_cores)  # Assuming your main function takes the number of cores as an argument
            end_time = time.time()
            elapsed_time = end_time - start_time
            times_for_this_core.append(elapsed_time)
        all_times.append(times_for_this_core)

    # Convert to numpy array for easier calculations
    all_times = np.array(all_times)
    mean_times = np.mean(all_times, axis=1)
    std_times = np.std(all_times, axis=1)

    plt.plot(core_values, mean_times, label="Mean Execution Time")
    plt.fill_between(core_values, mean_times - std_times, mean_times + std_times, color='gray', alpha=0.5, label="1 Std Deviation")
    plt.xlabel("Number of Cores")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Execution Time vs. Number of Cores")
    plt.legend()
    plt.show()