from torch_geometric.data import Data


class PairData(Data):
    def __init__(
        self, x_s, edge_index_s, edge_attr_s, x_t, edge_index_t, edge_attr_t, y
    ):
        super(PairData, self).__init__()

        if x_s is not None:
            self.x_s = x_s
        if edge_index_s is not None:
            self.edge_index_s = edge_index_s
        if edge_attr_s is not None:
            self.edge_attr_s = edge_attr_s

        if x_t is not None:
            self.x_t = x_t
        if edge_index_t is not None:
            self.edge_index_t = edge_index_t
        if edge_attr_t is not None:
            self.edge_attr_t = edge_attr_t

        if y is not None:
            self.y = y

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_s":
            return self.x_s.size(0)
        if key == "edge_index_t":
            return self.x_t.size(0)
        return super().__inc__(key, value, *args, **kwargs)

    def get_s(self) -> Data:
        return Data(self.x_s, self.edge_index_s, self.edge_attr_s, self.y)
        
    def get_t(self) -> Data:
        return Data(self.x_t, self.edge_index_t, self.edge_attr_t, self.y)

