import numpy as np
import matplotlib.pyplot as plt
import os
import pywt
import pywt.data
from quicktiff import TiffFile
from scipy.ndimage import filters
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.signal import argrelmax
import math
from scipy.signal import find_peaks
import tensorflow
import matplotlib.patches as patches


class HaarFilterBank(object):
    @property
    def filter_bank(self):
        x = np.arange(0,5,1)-2
        sigma = 1.5
        c = 2 / (np.sqrt(3 * sigma) * np.pi ** 1 / 4) * (1 - (x / sigma) ** 2) * np.exp(-x ** 2 / (2 * sigma ** 2))

        #c = math.sqrt(2)/2
        dec_lo, dec_hi, rec_lo, rec_hi = [0,0.0,1,0.0,0], [0,0.0,1.5,0.0,0], [0,0,0,0,0], c.tolist()
        #dec_lo, dec_hi, rec_lo, rec_hi = [c,c, c,c.c], [0,c/2, c,c/2,0], [-c,0, c], [c,0, -c]
        return [dec_lo, dec_hi, rec_lo, rec_hi]

filter_bank = HaarFilterBank()
myOtherWavelet = pywt.Wavelet(name="myHaarWavelet", filter_bank=filter_bank)


print(os.getcwd()+ r"\test_data\coordinate_recon_flim_n_big_tiff.tif")
tif = TiffFile(os.getcwd()+ r"\test_data\coordinate_recon_flim_n_big_tiff.tif")
original = tif.read_frame(1,0)[0]
original = original/original.max()*255

#test = stft(original, 1.5, window=(5,5))

# Load image
#original = pywt.data.camera()
HTHRESH=[70,150]
VTHRESH=[60,150]
HHTHRESH=[60,150]

# Wavelet transform of image, and plot approximation and details
titles = [ 'Approximation', ' Horizontal detail',
          'Vertical detail', "Diagonal detail",'Detected Localisations' ]

wavelet = pywt.DiscreteContinuousWavelet("db5")
#wavelet = pywt.scale2frequency(wavelet, 2)
coeffs2 = pywt.dwt2(original, wavelet)
LL, (LH, HL, HH) = coeffs2
LL[LL<170] = 0
LH[LH<HTHRESH[0]] = 0
LH[LH>HTHRESH[1]] = 0
HL[HL<VTHRESH[0]] = 0
HL[HL>VTHRESH[1]] = 0
HH[HH<HHTHRESH[0]] = 0
HH[HH>HHTHRESH[1]] = 0
coeffs2 = LL, (LH, HL, HH)
reconstruct = pywt.idwt2(coeffs2, wavelet)
reconstruct[reconstruct<50] = 0
neighborhood = generate_binary_structure(2, 2)

# apply the local maximum filter; all pixel of maximal value
# in their neighborhood are set to 1
local_max = filters.maximum_filter(reconstruct, footprint=neighborhood) == reconstruct
# local_max is a mask that contains the peaks we are
# looking for, but also the background.
# In order to isolate the peaks we must remove the background from the mask.

# we create the mask of the background
background = (reconstruct == 0)

# a little technicality: we must erode the background in order to
# successfully subtract it form local_max, otherwise a line will
# appear along the background border (artifact of the local maximum filter)
eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

# we obtain the final mask, containing only peaks,
# by removing the background from the local_max mask (xor operation)
detected_peaks = local_max ^ eroded_background
coords = np.array(np.where(detected_peaks!=0))
fig = plt.figure(figsize=(12, 3))
ax = fig.add_subplot(1, 6, 1)
ax.imshow(original, interpolation="nearest", cmap=plt.cm.gray)
for i in coords.T:
    rect = patches.Rectangle((i[1]-3, i[0]-3), 6, 6, linewidth=0.5, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.set_title("original", fontsize=10)

for i, a in enumerate([LL, LH, HL, HH, reconstruct]):
    ax = fig.add_subplot(1, 6, i + 2)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()
#todo: classify wavelet transform as localization or not localization