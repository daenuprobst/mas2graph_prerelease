from typing import Any, Iterable, List, Tuple, Optional
from molsetrep.encoders import Encoder
from matchms.Spectrum import Spectrum
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx
from matchms.Spectrum import Spectrum

from mas2graph.data import PairData


class GraphSpectrumEncoder(Encoder):
    def __init__(self) -> None: ...
    def get_graph(
        self,
        spectrum: Spectrum,
        include_precursor: bool = False,
        offset: int = 0,
        top_n: int = 0,
    ) -> nx.Graph:
        peaks = list(
            zip(
                list(map(float, spectrum.mz)),
                list(map(float, spectrum.intensities)),
            )
        )

        # Insert zero at beginning instead of precursor to not introduce bias towards it
        # Also: Makes better graphs when there is only one peak

        if top_n > 0:
            peaks.sort(key=lambda x: x[1])
            peaks = peaks[:top_n]

        peaks.append((0, 0))
        peaks.sort(key=lambda x: x[0])

        g = nx.Graph()

        mzs = []

        for i, (mz, intensity) in enumerate(peaks):
            mzs.append(mz)
            g.add_node(offset + i, mz=mz, intensity=intensity)

        for i in range(len(peaks) - 1):
            g.add_edge(offset + i, offset + i + 1, dist=mzs[i + 1] - mzs[i])

        if "precursor_mz" in spectrum.metadata and include_precursor:
            pid = offset + g.number_of_nodes()
            precursor_mz = spectrum.metadata["precursor_mz"]
            g.add_node(pid, mz=precursor_mz, intensity=2.0)

            for i in range(offset, pid):
                g.add_edge(i, pid, dist=abs(mzs[i - offset] - precursor_mz))

        return g

    def encode_pair(
        self,
        spectrum_a: Spectrum,
        spectrum_b: Spectrum,
        label: float,
        include_precursor: bool = False,
    ) -> Data:
        graph_a = self.get_graph(spectrum_a, include_precursor)
        graph_b = self.get_graph(
            spectrum_b, include_precursor, graph_a.number_of_nodes()
        )

        data_a = from_networkx(graph_a, group_node_attrs="all", group_edge_attrs="all")
        data_b = from_networkx(graph_b, group_node_attrs="all", group_edge_attrs="all")

        return PairData(
            data_a.x,
            data_a.edge_index,
            data_a.edge_attr,
            data_b.x,
            data_b.edge_index,
            data_b.edge_attr,
            y=torch.FloatTensor([float(label)]),
        )

    def encode_pairs(
        self,
        spectra: Iterable[Tuple[Spectrum, Spectrum]],
        labels: Iterable[float],
        include_precursor: bool = False,
    ) -> List[Data]:
        graphs = []
        for (spectrum_a, spectrum_b), label in zip(spectra, labels):
            graphs.append(
                self.encode_pair(spectrum_a, spectrum_b, label, include_precursor)
            )

        return graphs

    def encode(
        self,
        spectra: Iterable[Spectrum],
        labels: Iterable[float],
        include_precursor: bool = False,
        top_n: int = 0,
    ) -> List[Data]:
        graphs = []
        for spectrum, label in zip(spectra, labels):
            g = self.get_graph(
                spectrum, include_precursor=include_precursor, top_n=top_n
            )
            data = from_networkx(g, group_node_attrs="all", group_edge_attrs="all")
            data.y = label
            graphs.append(data)

        return graphs
