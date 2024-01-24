import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import filtfilt
import librosa
#import subprocess

"""EXTRACT AUDIO FILE FROM .WEBM FILE WITH FFMPEG"""
"""
# path to execute ffmpeg
ffmpeg_path = r'ffmpeg'

# convert .webm file to .wav file (extract audio from video file)
ffmpeg_command = [ffmpeg_path, '-i', 'drzepka_Silicone_medium_2023-12-02_15.15.51.webm', 'Silicone_medium_audio.wav']

# execute FFmpeg from PowerShell
try:
    subprocess.run(ffmpeg_command, check=True, shell=True)
    print("executing ffmpeg was successful")
except subprocess.CalledProcessError as e:
    print(f'Error while executing FFmpeg: {e}')
"""

"""DEFINE SAMPLES"""
sound1 = "drzepka_PU_medium_2023-12-02_15.19.47.wav"
synchronization_wave = "sync.wav"

# load samples
original_sound, sample_rate1 = librosa.load(sound1)
sync_wave, sample_rate = librosa.load(synchronization_wave)

""" visualize the original sound """
print(f'{synchronization_wave}: type {sync_wave.dtype}; Sample rate {sample_rate}')
print(f'{sound1}: type {original_sound.dtype}; Sample rate {sample_rate1}')

# calculate length of wave in seconds
length_in_s = sync_wave.shape[0] / sample_rate
length_in_s1 = original_sound.shape[0] / sample_rate1
print(f'{synchronization_wave} is {length_in_s} seconds')
print(f'{sound1} is {length_in_s1} seconds')

# determine x (time) achsis
time = np.arange(sync_wave.shape[0]) / sync_wave.shape[0] * length_in_s
time1 = np.arange(original_sound.shape[0]) / original_sound.shape[0] * length_in_s1
#make plot
plt.figure(figsize=(20, 10))
plt.title(f'{sound1} (orange) {synchronization_wave} (blue)')
plt.plot(time, sync_wave)
#plt.plot(time1, original_sound)
plt.show()

""" PREPROCESSING """
def highpass(x: np.ndarray, offset_samples: int, filter_order=200):
    h = np.ones(filter_order)/filter_order
    h = np.convolve(np.convolve(h, h), h)
    x = x[offset_samples:]
    y = np.zeros_like(x)
    for channel in range(x.shape[1]):
        y[:, channel] = x[:, channel] - filtfilt(h, 1, x[:, channel])
        y[:, channel] /= np.max(np.abs(y[:, channel]))
    return y

original_sound = highpass(original_sound[:,None], 0)[:,0]


""" NORMALIZE LOCALLY """
def correlate_by_sum(sync, input_array):
    correlated_window = np.array([])
    window = len(sync)
    print(f'{window} window')
    for i in range(len(input_array)- window + 1):
        # calculate sum of squares for the window & square root
        sum_of_squares = np.sum(np.square(input_array[i:i + window]))
        sqrt = np.sqrt(sum_of_squares)
        # use square root of sum of squares/window to normalise each value in the window
        normalized_values = (input_array[i: i + window]) / sqrt
        # correlate the sync wave and the normalized original sound
        correlated_value = np.sum(normalized_values * sync)
        correlated_window = np.append(correlated_window, correlated_value)
    # calculate maximal correlation and add window size do get correct index BUT in plot it is still wrong
    i = np.argmax(correlated_window) + window
    index_in_seconds = i / sample_rate1
    plt.title(f'original sound (red), correlation plot (blue)')
    plt.plot(correlated_window, color='blue')
    plt.plot( input_array, color='red')
    # plt.plot(sync, color='yellow')
    plt.show()
    max_corr_value = input_array[i]
    print(f'sync wave in original sound: index {i}, seconds {index_in_seconds}s')
    print(max_corr_value)

correlate_by_sum(sync_wave, original_sound)
