# from astropy.io import fits
# from astropy.io import ascii
# from astropy.table import vstack
# import os

# file_position = os.path.dirname(os.path.realpath(__file__))

# img = fits.getdata(os.path.join(file_position,"f444w_finalV4.fits"))[:2000,:2000]
# gth = fits.getdata(os.path.join(file_position,"f444w_finalV4.onlyPSF.fits"))[:2000,:2000]

# ILLUSTRIS DOCUMENTATION:
# https://archive.stsci.edu/hlsps/illustris/hlsp_illustris_hst-jwst-roman_multi_all_v1_readme.txt

import numpy as np
from astropy.io import fits
from astropy.table import Table
import os

file_position = os.path.dirname(os.path.realpath(__file__))

hdu = fits.open(
    os.path.join(
        file_position,
        "illustris_clean.fits",
    )
)

gth = hdu[3].data
cat = hdu[5]
cat = Table.read(cat)
cat = cat[["SubfindID", "new_j", "new_i", "halfmassrad_factor"]].to_pandas()
cat.columns = ["ID", "X", "Y", "R"]
cat["F"] = 0

np.random.seed(0)
noise = np.random.normal(loc=0.0, scale=0.2, size=gth.shape)

img = gth + noise

rms = np.zeros(gth.shape)
