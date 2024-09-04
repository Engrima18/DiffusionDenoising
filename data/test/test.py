import numpy as np
from astropy.io import fits
from astropy.table import Table
import os
import matplotlib.pyplot as plt

file_position = os.path.dirname(os.path.realpath(__file__))

gth = fits.open(
    os.path.join(
        file_position,
        "clean.fits",
    )
)[0].data

img = fits.open(
    os.path.join(
        file_position,
        "noisy.fits",
    )
)[0].data
