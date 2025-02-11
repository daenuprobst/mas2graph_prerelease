from typing import List, Tuple, Sequence
import numpy as np
import torch
from ms2deepscore.train_new_model import InchikeyPairGenerator
from torch_geometric.data import Data

from ms2deepscore.SettingsMS2Deepscore import (SettingsMS2Deepscore)
from mas2graph.data import PairData


class SpectrumPairGenerator(Sequence):
    """Generates data for training a siamese Pytorch model.

    This class provides a data generator specifically designed for training a Siamese Pytorch model with a curated set
    of compound pairs. It takes a InchikeyPairGenerator and randomly selects, augments and tensorizes spectra for each
    inchikey pair.

    By using pre-selected compound pairs (in the InchikeyPairGenerator), this allows more control over the training
    process. The selection of inchikey pairs does not happen in SpectrumPairGenerator (only spectrum selection), but in
    inchikey_pair_selection.py. In inchikey_pair_selection inchikey pairs are picked to balance selected pairs equally
    over different tanimoto score bins to make sure both pairs of similar and dissimilar compounds are sampled.
    In addition inchikeys are selected to occur equally for each pair.
    """

    def __init__(self, graph_list: List[Data],
                 inchikey_list: List[str],
                 selected_compound_pairs: InchikeyPairGenerator,
                 settings: SettingsMS2Deepscore):
        """Generates data for training a siamese Pytorch model.

        Parameters
        ----------
        spectrums
            List of matchms Spectrum objects.
        selected_compound_pairs
            SelectedCompoundPairs object which contains selected compounds pairs and the
            respective similarity scores.
        settings
            The available settings can be found in SettignsMS2Deepscore
        """
        self.current_batch_index = 0
        self.spectrums = graph_list
        self.spectrum_inchikeys = np.array(inchikey_list)

        # Set all other settings to input (or otherwise to defaults):
        self.model_settings = settings

        # Initialize random number generator
        if self.model_settings.use_fixed_set:
            if self.model_settings.shuffle:
                raise ValueError(
                    "The generator cannot run reproducibly when shuffling is on for `SelectedCompoundPairs`.")
            if self.model_settings.random_seed is None:
                self.model_settings.random_seed = 0
        self.rng = np.random.default_rng(self.model_settings.random_seed)

        unique_inchikeys = np.unique(self.spectrum_inchikeys)
        if len(unique_inchikeys) < self.model_settings.batch_size:
            raise ValueError("The number of unique inchikeys must be larger than the batch size.")
        self.fixed_set = {}

        self.selected_compound_pairs = selected_compound_pairs
        self.inchikey_pair_generator = self.selected_compound_pairs.generator(self.model_settings.shuffle, self.rng)
        self.nr_of_batches = len(unique_inchikeys)

    def __len__(self):
        return self.nr_of_batches

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_batch_index < self.nr_of_batches:
            batch = self.__getitem__(self.current_batch_index)
            self.current_batch_index += 1
            return batch
        self.current_batch_index = 0  # make generator executable again
        raise StopIteration

    def _spectrum_pair_generator(self):
        """Use the provided SelectedCompoundPairs object to pick pairs."""
        try:
            inchikey1, inchikey2, score = next(self.inchikey_pair_generator)
        except StopIteration as exc:
            raise RuntimeError("The inchikey pair generator is not expected to end, "
                               "but should instead generate infinite pairs") from exc

        graph_data_a = self._get_spectrum_with_inchikey(inchikey1)
        graph_data_b = self._get_spectrum_with_inchikey(inchikey2)

        graph_data = PairData(
            graph_data_a.x,
            graph_data_a.edge_index,
            graph_data_a.edge_attr,
            graph_data_b.x,
            graph_data_b.edge_index,
            graph_data_b.edge_attr,
            y=torch.FloatTensor([float(score)]),
        )
        return graph_data

    def __getitem__(self, batch_index: int):
        """Generate one batch of data.

        If use_fixed_set=True we try retrieving the batch from self.fixed_set (or store it if
        this is the first epoch). This ensures a fixed set of data is generated each epoch.
        """
        if self.model_settings.use_fixed_set and batch_index in self.fixed_set:
            return self.fixed_set[batch_index]
        if self.model_settings.random_seed is not None and batch_index == 0:
            self.rng = np.random.default_rng(self.model_settings.random_seed)
        spectrum_pairs = self._spectrum_pair_generator()
        return spectrum_pairs

    def _get_spectrum_with_inchikey(self, inchikey: str) -> Data:
        """
        Get a random spectrum matching the `inchikey` argument.

        NB: A compound (identified by an
        inchikey) can have multiple measured spectrums in a binned spectrum dataset.
        """
        matching_spectrum_id = np.where(self.spectrum_inchikeys == inchikey)[0]
        if len(matching_spectrum_id) <= 0:
            raise ValueError("No matching inchikey found (note: expected first 14 characters)")
        return self.spectrums[self.rng.choice(matching_spectrum_id)]
