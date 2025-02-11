from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Dataset


class SpecDataset(Dataset):
    def __init__(
        self,
        data_folder: Path,
        dataset_split: str,
        edge_attr: str = "None",
        random_walk: bool = True,
        shuffle: bool = False,
    ):
        super().__init__()

        self.data_folder = data_folder
        self.dataset_split = dataset_split
        self.edge_attr = edge_attr
        self.random_walk = random_walk
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # get all filenames in the folder
        self.files = [
            str(f) for f in Path(data_folder, dataset_split).glob("**/*") if f.is_file()
        ]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # shuffle files
        if shuffle:
            np.random.shuffle(self.files)

    def len(self):
        return len(self.files)

    def get(self, idx):
        # load the file
        file_name = self.files[idx]
        data = torch.load(self.files[idx])

        # # edge_attr
        if self.edge_attr == "None":
            data.edge_attr = None
        elif self.edge_attr == "Zero":
            data.edge_attr = torch.zeros(data.edge_index.shape[1], 2)

        # data.y, normalize by the sum of values
        # data.y = data.y / torch.nansum(data.y)

        return data, file_name
