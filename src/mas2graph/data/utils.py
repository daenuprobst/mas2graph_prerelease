import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx


def spectra_to_graph(peaks: list[tuple[float, float]]) -> nx.Graph:
    peaks_ = sorted(peaks, key=lambda x: x[0])
    g = nx.Graph()

    mzs = []

    for i, (mz, intensity) in enumerate(peaks_):
        mzs.append(mz)
        g.add_node(i, mz=mz, intensity=intensity)

    for i in range(len(peaks_) - 1):
        g.add_edge(i, i + 1, dist=mzs[i + 1] - mzs[i])

    return g


def spectra_to_pyg(peaks: list[tuple[float, float]]) -> Data:
    g = spectra_to_graph(peaks)
    return from_networkx(g, group_node_attrs="all", group_edge_attrs="all")
