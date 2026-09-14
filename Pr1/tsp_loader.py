import os

import numpy as np
from tsp import TSP


class TSPLoader:
    def __init__(self, dir_name) -> None:
        self.dir_name = dir_name
        self.files = []
        self.idx = 0

    def parse(self) -> TSP:
        file = self.files[self.idx]
        line_idx = 0
        data = []

        ret: TSP = TSP()
        for line in file:
            if line_idx > 6 and line.find("EOF") == -1:
                s = line.split()
                data.append((float(s[1]), float(s[2])))

            s = line.split(":")
            key = s[0].strip()

            if key.find("NAME") != -1:
                ret.name = s[-1].strip()
            elif key.find("COMMENT") != -1:
                ret.comment = s[-1].strip()
            elif key.find("DIMENSION") != -1:
                ret.dimension = s[-1].strip()
            elif key.find("EDGE_WEIGHT_TYPE") != -1:
                ret.edge_weight_type = s[-1].strip()
            elif key.find("TYPE") != -1:
                ret.type = s[-1].strip()
            line_idx += 1

        self.idx += 1
        ret.datapoints = np.array(data)
        return ret

    def load(self):
        paths = os.listdir(self.dir_name)
        self.files = [open(self.dir_name + "/" + path, "r") for path in paths]
