import numpy as np


class TSP:
    def __init__(self) -> None:
        self.name = ""
        self.comment = ""
        self.type = ""
        self.dimension = ""
        self.edge_weight_type = ""
        self.datapoints = np.array([], dtype=float)
