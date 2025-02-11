import pickle
from pathlib import Path
from typing import Iterable
import typer
import torch
from torch.utils.data import TensorDataset
from matchms.importing import load_from_mgf
from matchms.Spectrum import Spectrum
from tqdm import tqdm

from rdkit.Chem import MolFromSmiles
from rdkit.Chem.QED import qed

from torch_geometric.data import Data

from mas2graph.data import SpectrumEncoder, BinnedSpectrumEncoder, GraphSpectrumEncoder

app = typer.Typer(pretty_exceptions_show_locals=False)


def bin_spectrum(
    spectra: Iterable[Spectrum],
    y: Iterable[float],
    n_bins: int = 10000,
    include_precursor: bool = True,
) -> TensorDataset:
    encoder = BinnedSpectrumEncoder(n_bins)
    all_peaks = []

    for spectrum in spectra:
        peaks = list(
            zip(list(map(float, spectrum.mz)), list(map(float, spectrum.intensities)))
        )

        # Insert zero at beginning instead of precursor to not introduce bias towards it
        # Also: Makes better graphs when there is only one peak
        peaks.append((0, 0))

        if include_precursor and "precursor_mz" in spectrum.metadata:
            peaks.append((spectrum.metadata["precursor_mz"], 2.0))

        # Make sure that the list is sorted by m/z
        peaks.sort(key=lambda x: x[0])

        all_peaks.append(peaks)

    # This method transforms the list of m/z, intensity tuples and turns it into an pyg data object (graph)
    return encoder.encode(all_peaks, y, torch.float32)


# to use the train_generator
@app.command()
def main(mgf_path: Path, out_path: Path):
    # It assumes that all spectra are annotated with valid smiles (parsible by rdkit) and inchikeys.
    spectra = list(
        tqdm(
            load_from_mgf(str(mgf_path)),
            desc="loading spectra",
        )
    )

    encoder = SpectrumEncoder()
    graph_encoder = GraphSpectrumEncoder()

    y = []
    # graph_data = []

    for spectrum in tqdm(spectra, desc="Calculating QED"):
        y_tmp = qed(MolFromSmiles(spectrum.metadata["smiles"]))
        # data = spectrum_to_graph(spectrum)
        # data.y = y_tmp

        y.append(y_tmp)
        # graph_data.append(data)

    set_data = encoder.encode(spectra, y, torch.float32)
    binned_data = bin_spectrum(spectra, y)
    graph_data = graph_encoder.encode(spectra, y, True, 100)

    with open(out_path, "wb") as f:
        pickle.dump(
            {
                "graph_data": graph_data,
                "set_data": set_data,
                "binned_data": binned_data,
                "y": y,
            },
            f,
        )


if __name__ == "__main__":
    app()
