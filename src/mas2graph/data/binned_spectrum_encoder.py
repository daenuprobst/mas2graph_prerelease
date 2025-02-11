from typing import Any, Iterable, Optional
import numpy as np
from molsetrep.encoders import Encoder
from matchms.Spectrum import Spectrum
from torch.utils.data import TensorDataset
from scipy.stats import binned_statistic
import torch


class BinnedSpectrumEncoder(Encoder):
    def __init__(self, n_bins=10000) -> None:
        super().__init__("SpectrumEncoder")
        self.n_bins = n_bins

    def encode(
        self,
        spectra: Iterable[Any],
        labels: Iterable[float],
        label_dtype: Optional[torch.dtype],
        **kwargs,
    ) -> TensorDataset:
        peaks = []

        for spectrum in spectra:
            x = np.array(spectrum).T
            binned_peaks = binned_statistic(
                x[0],
                x[1],
                bins=self.n_bins,
                statistic="max",
                range=(0.0, 1000.0),
            )

            x = np.nan_to_num(binned_peaks.statistic)
            peaks.append(x)

        return TensorDataset(
            torch.FloatTensor(np.array(peaks)), torch.tensor(labels, dtype=label_dtype)
        )
