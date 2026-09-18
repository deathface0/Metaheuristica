import time
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

        return tour, int(total_cost)

    def run(self):
        for alg in self.data.algorithms:
            costs = []
            times = []

            for seed in self.data.seeds:
                t0 = time.perf_counter()
                res = None

                if alg == "gre":
                    res = self.tsp_greedy(seed=seed)
                elif alg == "gra":
                    pass
                elif alg == "bl":
                    pass
                elif alg == "tabu":
                    pass
                else:
                    pass

                t1 = time.perf_counter()
                elapsed = t1 - t0

                if res is not None:
                    tour, cost = res
                    costs.append(cost)
                    times.append(elapsed)
                    print(f"[INFO] Algoritmo: {alg} | Seed: {seed} | Coste: {cost} | Tiempo: {elapsed:.4f}s")

            if costs:
                print(f"[INFO] Resumen {alg} | Min: {min(costs)} | Media: {np.mean(costs):<10.2f} | Desv: {np.std(costs):<8.2f} | Tiempo medio: {np.mean(times):.4f}s")
            else:
                print(f"[WARN] Algoritmo '{alg}' no implementado.")
