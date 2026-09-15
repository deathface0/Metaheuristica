import os
import numpy as np
from tsp import TSP


class TSPLoader:
    def __init__(self, dir_name: str) -> None:
        self.dir_name = dir_name
        self.file_paths = []
        self.idx = 0

    def length(self) -> int:
        return len(self.file_paths)

    def load(self):
        self.file_paths = [
            os.path.join(self.dir_name, path)
            for path in sorted(os.listdir(self.dir_name))
            if path.endswith(".tsp")
        ]
        self.idx = 0

    def parse(self) -> TSP:
        file_path = self.file_paths[self.idx]
        self.idx += 1

        ret = TSP()
        data = []
        in_coord_section = False

        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("EOF"):
                    break

                if in_coord_section:
                    parts = line.split()
                    if len(parts) >= 3:
                        data.append((float(parts[1]), float(parts[2])))
                    continue

                if line.startswith("NODE_COORD_SECTION"):
                    in_coord_section = True
                    continue

                if ":" in line:
                    key, val = line.split(":", 1)
                    key = key.strip().upper()
                    val = val.strip()

                    if key == "NAME":
                        ret.name = val
                    elif key == "COMMENT":
                        ret.comment = val
                    elif key == "DIMENSION":
                        ret.dimension = int(val)
                    elif key == "EDGE_WEIGHT_TYPE":
                        ret.edge_weight_type = val
                    elif key == "TYPE":
                        ret.type = val

        ret.datapoints = np.array(data, dtype=float)
        return ret

