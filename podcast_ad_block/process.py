#! /usr/bin/env python3

import pydub
from pydub import AudioSegment
from pathlib import Path
import numpy as np
import scipy
import scipy.signal
from podcast_ad_block.config import Config


DOWNSAMPLE = 1_000


def process_file(audio_path: str | Path, config: Config) -> list[tuple[float, float]]:
    audio: AudioSegment = AudioSegment.from_file(audio_path)
    ch = audio.channels
    marker_path = config.ad_songs[0]  # TODO
    marker: AudioSegment = AudioSegment.from_file(marker_path)
    assert marker.channels == ch,  "Audio and marker must have same number of channels"
    assert marker.frame_rate == audio.frame_rate, "Audio and marker must have same sample rate"

    corr = do_correlation(audio, marker, downsample=DOWNSAMPLE)
    sample_freq = audio.frame_rate / DOWNSAMPLE
    ads = find_ads(corr, sample_freq, config)
    return ads


def find_ads(corr: np.ndarray, sample_freq: float, config: Config) -> list[tuple[float, float]]:
    start: int = int(config.ignore_first_seconds * sample_freq)
    end: int = len(corr) - int(config.ignore_last_seconds * sample_freq)

    corr_truncated = corr[start:end]
    corr_truncated[corr_truncated < corr_truncated.max() / 2] = 0

    peaks, _ = scipy.signal.find_peaks(
        corr_truncated, distance=int(config.minimum_ad_length_seconds * sample_freq))
    peaks += start

    spacing = peaks[1:] - peaks[:-1]
    ad_start_indexes = np.argwhere(
        spacing < config.maximum_ad_length_seconds * sample_freq).reshape((-1,))
    ad_starts = peaks[ad_start_indexes] + config.ad_buffer_seconds * sample_freq
    ad_ends = peaks[ad_start_indexes + 1] - config.ad_buffer_seconds * sample_freq

    def to_seconds(x): return x / sample_freq
    out = list(zip(map(to_seconds, ad_starts), map(to_seconds, ad_ends)))
    return out


def do_correlation(audio_file: AudioSegment, marker_file: AudioSegment, window: int = 1_000_000, downsample: int = 1_000):
    audio = np.array(audio_file.get_array_of_samples()
                     ).reshape((-1, audio_file.channels))
    marker = np.array(marker_file.get_array_of_samples()
                      ).astype(np.int64).reshape((-1, marker_file.channels))

    pad_start = (marker.shape[0] - 1) // 2
    pad_end = (marker.shape[0] + 1) // 2
    buffer_size = window + pad_start + pad_end
    audio_buffer = np.zeros(
        dtype=np.int64, shape=(buffer_size, audio.shape[1]))
    segments = ((audio.shape[0] - 1) // window) + 1
    corr = np.zeros(dtype=np.int64, shape=(segments, window // downsample))
    for i in range(0, segments):
        # print(f"Segment {i}/{segments}")
        start = i * window - pad_start
        end = start + pad_start + window + pad_end
        if start < 0:
            audio_buffer[-start:, :] = audio[0:end, :]
        elif end >= audio.shape[0]:
            buffer_end = audio.shape[0] - start
            audio_buffer[:buffer_end, :] = audio[start:, :]
            audio_buffer[buffer_end:, :] = 0
        else:
            audio_buffer[:, :] = audio[start:end, :]
        corr[i, :] = process_segment(
            audio_buffer, marker, downsample=downsample)

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
