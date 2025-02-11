from matchms.importing import load_from_mgf
import pickle
from tqdm import tqdm
from torch_geometric.data import Data
from mas2graph.data.utils import spectra_to_pyg
from matchms.Spectrum import Spectrum


def get_mz_intensity_pair(spectrum: Spectrum, include_precursor: bool = True) -> Data:
    peaks = list(
        zip(list(map(float, spectrum.mz)), list(map(float, spectrum.intensities)))
    )

    # Insert zero at beginning instead of precursor to not introduce bias towards it
    # Also: Makes better graphs when there is only one peak
    peaks.append((0, 0))

    if include_precursor and "precursor_mz" in spectrum.metadata:
        peaks.append((spectrum.metadata["precursor_mz"], 1.0))

    # Make sure that the list is sorted by m/z
    peaks.sort(key=lambda x: x[0])

    # This method transforms the list of m/z, intensity tuples and turns it into an pyg data object (graph)
    return spectra_to_pyg(peaks)


def convert_spectra_to_graph(spectrum_generator):
    graph_list = []
    inchikey_list = []
    for spectrum in tqdm(spectrum_generator, desc="convert spectra to graphs"):
        graph_data = get_mz_intensity_pair(spectrum)
        graph_list.append(graph_data)
        inchikey_list.append(spectrum.get("inchikey")[:14])
    return graph_list, inchikey_list


def create_processed_spectra_file(spectrum_file_name, graph_file_name):
    spectrum_generator = load_from_mgf(spectrum_file_name)
    graph_list = convert_spectra_to_graph(spectrum_generator)
    with open(graph_file_name, "wb") as f:
        pickle.dump(graph_list, f)


if __name__ == "__main__":
    create_processed_spectra_file("../../../data/raw_spectra/negative_training_spectra.mgf",
                                  "../data/processed_spectra/negative_training_spectra_graphs.pickle")
    create_processed_spectra_file("../../../data/raw_spectra/negative_testing_spectra.mgf",
                                  "../data/processed_spectra/negative_testing_spectra_graphs.pickle")
    create_processed_spectra_file("../data/raw_spectra/negative_validation_spectra.mgf",
                                  "../data/processed_spectra/negative_validation_spectra_graphs.pickle")

