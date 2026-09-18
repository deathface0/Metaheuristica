import numpy as np

class TSP():
    def __init__(self, seeds, algs, params) -> None:
        self.seeds = seeds
        self.algorithms = algs
        self.params = params

        self.name = ""
        self.comment = ""
        self.type = ""
        self.dimension = 0
        self.edge_weight_type = ""
        self.datapoints = np.array([], dtype=float)
        self.dist = None

    def calc_dist(self, chunk_size: int = 1024):
        n = len(self.datapoints)
        if n == 0:
            self.dist = np.empty((0, 0), dtype=int)
            return self.dist

        self.dist = np.empty((n, n), dtype=int)

        x = self.datapoints[:, 0]
        y = self.datapoints[:, 1]

        for i in range(0, n, chunk_size):
            dx = x[i:i + chunk_size, np.newaxis] - x
            dy = y[i:i + chunk_size, np.newaxis] - y
            d = np.sqrt(dx * dx + dy * dy)
            self.dist[i:i + chunk_size] = np.round(d)

        return self.dist

        

    
