import numpy as np

class TSPSolver:
    def __init__(self, tsp_data):
        self.data = tsp_data

    def tsp_greedy(self, seed=0):
        np.random.seed(seed)
        num_nodes = len(self.data.dist)
        start_node = np.random.randint(num_nodes)
        
        visited = np.zeros(num_nodes, dtype=bool)
        tour = [start_node]
        visited[start_node] = True
        total_cost = 0
        current_node = start_node

        for _ in range(num_nodes - 1):
            distances = self.data.dist[current_node].astype(float)
            distances[visited] = np.inf
            
            next_node = np.argmin(distances)
            min_distance = distances[next_node]

            tour.append(int(next_node))
            visited[next_node] = True
            total_cost += min_distance
            current_node = next_node

        total_cost += self.data.dist[current_node][start_node]
        tour.append(start_node)

        return tour, total_cost

    def run(self):
        for alg in self.data.algorithms:
            for seed in self.data.seeds:
                print(f"Corriendo algoritmo '{alg}' con seed '{seed}'")
                if alg == "gre":
                    res = self.tsp_greedy(seed=seed)
                    print(f" -> Coste obtenido: {res[1]}")
                elif alg == "gra":
                    pass
                elif alg == "bl":
                    pass
                elif alg == "tabu":
                    pass
                else:
                    pass
