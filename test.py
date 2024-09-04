#! /usr/bin/env python3

import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
from podcast_ad_block.process import process_file
from podcast_ad_block.config import load_config


DATA_DIR = "./data/test"
MARKER_PATH = "./data/my_favorite_podcast/ad_song.mp3"
DOWNSAMPLE = 1000


def seconds_to_timestamp(sec: float | int):
    sec = int(sec)
    ts = f"{((sec % 3600) // 60):02d}:{(sec % 60):02d}"
    if sec > 3600:
        ts = f"{sec // 3600}:{ts}"
    return ts


def batch_process():
    config = load_config("./config.json")[0]

    dirpath, _, filenames = next(os.walk(DATA_DIR))
    for filename in filenames:
        file = os.path.join(dirpath, filename)
        print(f"Processing {file}")
        ads = process_file(file, config)
        ads = [(seconds_to_timestamp(a), seconds_to_timestamp(b)) for a, b in ads]
        print(f"File: {filename}, Ads: {ads}")


if __name__ == "__main__":
    batch_process()
