from scipy.ndimage import filters
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
import numpy as np
import pywt


class Binning():
    HTHRESH = [80, 150]
    VTHRESH = [70, 150]
    HHTHRESH = [60, 150]
    def __init__(self):
        self.wavelet = pywt.DiscreteContinuousWavelet("db5")

    def filter(self, original, wavelet=None):
        if wavelet is not None:
            self.wavelet = wavelet
        coeffs2 = pywt.dwt2(original, self.wavelet)
        LL, (LH, HL, HH) = coeffs2
        LL[LL < 170] = 0
        LH[LH < self.HTHRESH[0]] = 0
        LH[LH > self.HTHRESH[1]] = 0
        HL[HL < self.VTHRESH[0]] = 0
        HL[HL > self.VTHRESH[1]] = 0
        HH[HH < self.HHTHRESH[0]] = 0
        HH[HH > self.HHTHRESH[1]] = 0
        coeffs2 = LL, (LH, HL, HH)
        reconstruct = pywt.idwt2(coeffs2, self.wavelet)
        reconstruct[reconstruct < 50] = 0
        return reconstruct

    def get_coords(self, reconstruct):
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
        coords = np.array(np.where(detected_peaks != 0))
        return coords.T

