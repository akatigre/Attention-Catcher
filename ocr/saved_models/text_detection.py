import pandas as pd
import os
from PIL import Image
from text_detection.test import *
os.chdir('./saved_models')
!wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-VYCTLAlUfgdTDuXZB1CSu1b5Pbtclml' -O text_detection.pth
os.chdir('..')
!python ./text_detection/test.py \
--trained_model ./saved_models/text_detection.pth \
--test_folder ./text_detection/test/
