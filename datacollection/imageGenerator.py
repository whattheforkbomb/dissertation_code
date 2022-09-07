# Read in the yuv files and create the RGB and greyscale PNGs required.

import sys
from pathlib import Path
from sre_parse import HEXDIGITS
import numpy as np
import cv2
import csv

WIDTH = 1280
HEIGHT = 720
SYNCED_DATA_PATH = 'synced_data'

root_dir = Path(sys.argv[1])
for participant_path in root_dir.iterdir():
    data_path = participant_path / SYNCED_DATA_PATH
    print(f"Participant: {participant_path.name}")
    for csv_path in data_path.iterdir():
        with open(csv_path, newline='') as csv_file:
            print(f"Generating RGB Images for: {csv_path.stem}")
            reader = csv.reader(csv_file)
            header = reader.__next__()
            previous_img_yuv_path = ''
            for row in reader:
                img_yuv_path = "{}.yuv".format(row[2][2:])
                if img_yuv_path == previous_img_yuv_path: # don't need to process twice
                    continue
                # Read and convert image
                with open(participant_path / img_yuv_path, 'rb') as img_path:
                    img_bytes = img_path.read()
                img_arr = np.frombuffer(img_bytes, np.uint8).reshape((HEIGHT*3)//2, WIDTH)
                rgb = cv2.cvtColor(img_arr, cv2.COLOR_YUV2RGB_NV21)
                # rotate image
                # rgb_vertical = np.rot90(rgb)
                rgb_file = data_path / "{}png".format(img_yuv_path[:-3])
                rgb_file.parents[0].mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(rgb_file).replace('\\\\', '/'), rgb)
