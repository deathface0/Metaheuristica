import time
import numpy as np

class TSPSolver:
    def __init__(self, tsp_data):
        self.data = tsp_data

    def tsp_greedy(self):
        n = len(self.data.dist)

        # Buscamos la ciudad mas centrica
        start_node = int(np.argmin(self.data.dist.sum(axis=1)))
        
        tour = np.empty(n, dtype=np.int32)
        tour[0] = start_node
        
        visited = np.zeros(n, dtype=bool)
        visited[start_node] = True
        current_node = start_node

        # Buscamos el vecino mas cercano
        for i in range(1, n):
            distances = self.data.dist[current_node].astype(float)
            distances[visited] = np.inf
            next_node = int(np.argmin(distances))
            
            tour[i] = next_node
            visited[next_node] = True
            current_node = next_node
            
        path_cost = np.sum(self.data.dist[tour[:-1], tour[1:]])
        return_cost = self.data.dist[tour[-1], tour[0]]
        total_cost = int(path_cost + return_cost)
        
        return tour.tolist(), total_cost

    def tsp_random_greedy(self, seed: int):
        np.random.seed(seed)

        n = len(self.data.dist)
        k = self.data.params["k"]
        
        ordered_nodes = np.argsort(self.data.dist.sum(axis=1)).tolist()
        tour = np.empty(n, dtype=np.int32)

        for i in range(n):
            k = min(k, len(ordered_nodes))
            rand_idx = np.random.randint(0, k)
            tour[i] = ordered_nodes.pop(rand_idx)

        path_cost = np.sum(self.data.dist[tour[:-1], tour[1:]])
        return_cost = self.data.dist[tour[-1], tour[0]]
        total_cost = int(path_cost + return_cost)

        return tour.tolist(), total_cost 

    def _deltas_intercambio(self, s, i):
        d = self.data.dist
        n = len(s)
        ant, sig = np.roll(s, 1), np.roll(s, -1)
        a = s[i]

        desaparecen = d[ant[i], a] + d[a, sig[i]] + d[ant, s] + d[s, sig]
        nuevos = d[ant[i], s] + d[sig[i], s] + d[a, ant] + d[a, sig]
        delta = nuevos - desaparecen

        # Ciudades contiguas: el arco entre ellas se ha restado dos veces
        contiguas = [(i - 1) % n, (i + 1) % n]
        delta[contiguas] += 2 * d[a, s[contiguas]]
        delta[i] = 0

        return delta

    def busqueda_local(self, seed: int, max_iter: int = None):
        if max_iter is None:
            max_iter = self.data.params.get("bl_iter", 10000)

        # Solucion inicial
        tour, coste = self.tsp_random_greedy(seed)
        s = np.array(tour)
        n = len(s)

        dlb = np.zeros(n, dtype=bool)
        i = np.random.randint(n)
        it = 0

        # Sin ciudades prometedoras ningun vecino mejora: fin de la busqueda
        while it < max_iter and not dlb.all():
            if not dlb[i]:
                delta = self._deltas_intercambio(s, i)
                orden = (i + 1 + np.arange(n - 1)) % n  # j circular desde i+1
                mejoras = orden[delta[orden] < 0]

                if mejoras.size:
                    j = mejoras[0]  # Primer mejor
                    s[[i, j]] = s[[j, i]]
                    coste += delta[j]
                    dlb[[i, j]] = False
                    it += 1
                else:
                    dlb[i] = True

            i = (i + 1) % n

        return s.tolist(), int(coste)

    def run(self):
        for alg in self.data.algorithms:
            costs = []
            times = []

            for seed in self.data.seeds:
                t0 = time.perf_counter()
                res = None

                if alg == "gre":
                    res = self.tsp_greedy()
                elif alg == "gra":
                    res = self.tsp_random_greedy(seed)
                elif alg == "bl":
                    res = self.busqueda_local(seed)
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
