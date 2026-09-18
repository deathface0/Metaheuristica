import numpy as np

class TSP():
    def __init__(self) -> None:
        self.name = ""
        self.comment = ""
        self.type = ""
        self.dimension = 0
        self.edge_weight_type = ""
        self.datapoints = np.array([], dtype=float)
        self.dist = None

    def calc_dist(self, round_int: bool = True):
        x = self.datapoints[:, 0]
        y = self.datapoints[:, 1]
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(dx * dx + dy * dy)

        if round_int or self.edge_weight_type.upper() == "EUC_2D":
            self.dist = np.round(dist).astype(int)
        else:
            self.dist = dist
        return self.dist

        

    
