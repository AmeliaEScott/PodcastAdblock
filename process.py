#! /usr/bin/env python3

import pydub
from pathlib import Path
import numpy as np
import scipy
import scipy.signal
import matplotlib.pyplot as plt


def process_file(audio_path: str | Path, marker_path: str | Path, window: int = 1_000_000, downsample: int = 1_000):
    audio = pydub.AudioSegment.from_file(audio_path)
    ch = audio.channels
    audio = np.array(audio.get_array_of_samples()).reshape((-1, ch))
    marker = pydub.AudioSegment.from_file(marker_path)
    assert marker.channels == ch,  "Audio and marker must have same number of channels"
    marker = np.array(marker.get_array_of_samples()).astype(np.int64).reshape((-1, ch))
    print(f"Audio shape: {audio.shape}, Marker shape: {marker.shape}")
    
    
    pad_start = (marker.shape[0] - 1) // 2
    pad_end = (marker.shape[0] + 1) // 2
    buffer_size = window + pad_start + pad_end
    audio_buffer = np.zeros(dtype=np.int64, shape=(buffer_size, ch))
    segments = ((audio.shape[0] - 1) // window) + 1
    corr = np.zeros(dtype=np.int64, shape=(segments, window // downsample))
    for i in range(0, segments):
        print(f"Segment {i}/{segments}")
        start = i * window - pad_start
        end = start + pad_start + window + pad_end
        if start < 0:
            print("Start")
            audio_buffer[-start:, :] = audio[0:end, :]
        elif end >= audio.shape[0]:
            print("End")
            buffer_end = audio.shape[0] - start
            audio_buffer[:buffer_end, :] = audio[start:, :]
            audio_buffer[buffer_end:, :] = 0
        else:
            print("Middle")
            audio_buffer[:, :] = audio[start:end, :]
        corr[i, :] = process_segment(audio_buffer, marker, downsample=downsample)
    
    return corr.reshape((-1,))




def process_segment(audio: np.ndarray, marker: np.ndarray, downsample: int):
    """Process a segment of audio.
    
    Perform the following steps:
     1. For each channel, correlate `audio` with `marker`
     2. Sum all channels to form a 1-channel correlation
     3. Downsample the correlation by finding the maximum correlation in each window

    Args:
        audio (np.ndarray): Must be of shape `(samples, num_channels)`
        marker (np.ndarray): Must be of shape `(samples, num_channels)`, where `num_channels` matches `audio`.
        downsample (int): Window size for downsampling. For example, a value of 100 will result in an array 100x smaller.

    Returns:
        np.ndarray: A 1-D array of correlation values, downsampled according to `downsample`.
    """
    assert len(audio.shape) == 2, "audio must be 2-dimensional"
    assert len(marker.shape) == 2, "marker must be 2-dimensional"
    assert audio.shape[1] == marker.shape[1], "audio and marker must have same number of channels"
    channels = audio.shape[1]

    # corr = np.stack([scipy.signal.correlate(audio[:, i], marker[:, i]) for i in range(0, channels)], axis=1)
    # # corr.shape == [num_samples, channels]
    # assert len(corr.shape) == 2 and corr.shape[1] == audio.shape[1]
    # corr: np.ndarray = corr.sum(axis=1)
    # assert len(corr.shape) == 1
    corr = scipy.signal.correlate(audio, marker, mode="valid")
    corr = corr.reshape((-1, downsample))
    corr = corr.max(axis=1)
    assert len(corr.shape) == 1
    return corr
    


def main():
    pass

if __name__ == "__main__":
    corr = process_file("./data/my_favorite_podcast/test.mp3","./data/my_favorite_podcast/ad_song.mp3")
    plt.plot(corr)
    plt.show()