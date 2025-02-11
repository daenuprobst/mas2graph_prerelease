from typing import Any, Iterable, Optional
from molsetrep.encoders import Encoder
from matchms.Spectrum import Spectrum
from torch.utils.data import TensorDataset
import torch


class SpectrumEncoder(Encoder):
    def __init__(self) -> None:
        super().__init__("SpectrumEncoder")

    def encode(
        self,
        spectra: Iterable[Spectrum],
        labels: Iterable[float],
        label_dtype: Optional[torch.dtype],
        **kwargs,
    ) -> TensorDataset:
        peaks = []

        for spectrum in spectra:
            peaks.append(
                list(
                    zip(
                        list(map(float, spectrum.mz)),
                        list(map(float, spectrum.intensities)),
                    )
                )
            )

        return super().to_multi_tensor_dataset([peaks], labels, label_dtype)
