import json
import os
import time
from collections import deque

import numpy as np

INT32_MAX = np.iinfo(np.int32).max


class TSPSolver:
    def __init__(self, tsp_data):
        self.data = tsp_data

    def log_to_file(self, curr_alg, cost, params, iter: int, seed=None):
        to_log = {
            "alg": self.data.name,
            "params": params,
            "total_iter": iter,
            "total_cost": cost,
        }
        os.makedirs("./logs", exist_ok=True)
        filename = f"./logs/{self.data.name}_{curr_alg}.json.log"
        if seed:
            filename = f"./logs/{self.data.name}_{curr_alg}_{seed}.json.log"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(to_log, f, indent=4, ensure_ascii=False, default=int)

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

        self.log_to_file("gre", total_cost, {}, n)
        return tour.tolist(), total_cost

    def tsp_random_greedy(self, seed: int):
        np.random.seed(seed)

        n = len(self.data.dist)
        k = self.data.params["k"]

        ordered_nodes = np.argsort(self.data.dist.sum(axis=1)).tolist()
        tour = np.empty(n, dtype=np.int32)

        for i in range(n):
            kk = min(k, len(ordered_nodes))
            rand_idx = np.random.randint(0, kk)
            tour[i] = ordered_nodes.pop(rand_idx)

        path_cost = np.sum(self.data.dist[tour[:-1], tour[1:]])
        return_cost = self.data.dist[tour[-1], tour[0]]
        total_cost = int(path_cost + return_cost)

        self.log_to_file("gra", total_cost, {"k": k, "seed": seed}, n, seed)
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

    def busqueda_local(self, seed: int):
        max_iter = self.data.params.get("bl_iter", 10000)
        if max_iter <= 0:
            raise ValueError("Las iteraciones y la tenencia deben ser positivas")
        # Solucion inicial
        tour, coste = self.tsp_random_greedy(seed)
        start = coste
        s = np.array(tour)
        n = len(s)

        dlb = np.zeros(n, dtype=bool)
        i = np.random.randint(n)
        it: int = 0

        # Sin ciudades prometedoras ningun vecino mejora: fin de la busqueda
        while it < max_iter and not dlb.all():
            if not dlb[i]:
                delta = self._deltas_intercambio(s, i)
                orden = (i + 1 + np.arange(n - 1)) % n  # j circular desde i+1
                mejoras = orden[delta[orden] < 0]

                if mejoras.size:
                    j = mejoras[0]  # Primer mejor
                    s[[i, j]] = s[[j, i]]
                    coste += int(delta[j])
                    dlb[[i, j]] = False
                    it += 1
                else:
                    dlb[i] = True
            i = (i + 1) % n

        self.log_to_file(
            "bl",
            coste,
            {"start": start, "bl_iter": max_iter, "seed": seed},
            int(it),
            seed,
        )
        return s.tolist(), int(coste)

    def _deltas_filas(self, s, ant, sig, arcos, P):
        """Deltas de intercambiar cada posición de P con todas las posiciones."""
        d = self.data.dist
        n = len(s)
        a = s[P]

        filas_a = d[a]
        nuevos = (
            np.take(d[ant[P]], s, axis=1)
            + np.take(d[sig[P]], s, axis=1)
            + np.take(filas_a, ant, axis=1)
            + np.take(filas_a, sig, axis=1)
        )
        delta = nuevos - arcos[P, np.newaxis] - arcos

        # Ciudades contiguas: el arco entre ellas se ha restado dos veces
        r = np.arange(len(P))
        delta[r, (P + 1) % n] += 2 * d[a, sig[P]]
        delta[r, (P - 1) % n] += 2 * d[a, ant[P]]

        # Intercambiar una posición consigo misma no es un movimiento
        delta[r, P] = INT32_MAX

        return delta

    def _construir_deltas(self, s, delta):
        """Rellena delta con todos los intercambios de s, por bloques de filas.

        Devuelve las ciudades anterior y siguiente de cada posición, el coste
        de sus dos arcos y el mínimo (con su columna) de cada fila.
        """
        d = self.data.dist
        n = len(s)

        ant, sig = np.roll(s, 1), np.roll(s, -1)
        arcos = d[ant, s] + d[s, sig]

        fila_min = np.empty(n, dtype=np.int32)
        fila_arg = np.empty(n, dtype=np.intp)

        bloque = max(1, 2**20 // n)
        for p0 in range(0, n, bloque):
            p1 = min(p0 + bloque, n)
            filas = self._deltas_filas(s, ant, sig, arcos, np.arange(p0, p1))
            delta[p0:p1] = filas
            fila_arg[p0:p1] = filas.argmin(axis=1)
            fila_min[p0:p1] = filas[np.arange(p1 - p0), fila_arg[p0:p1]]

        return ant, sig, arcos, fila_min, fila_arg

    def _tabu_oscilacion(self, mlp, rng, intensificar, escala):
        """Construye una solución greedy guiada por las frecuencias de la MLP.

        Intensificar sigue el arco más frecuente y diversificar el menos
        frecuente. Los empates se deshacen por distancia: escala es mayor que
        cualquier distancia, así que frecuencia y distancia van en un entero.
        """
        d = self.data.dist
        n = len(d)

        solucion = np.empty(n, dtype=np.intp)
        actual = int(rng.integers(n))
        solucion[0] = actual

        # Las ciudades no visitadas están en restantes[:n - posicion]
        restantes = np.delete(np.arange(n), actual)

        for posicion in range(1, n):
            m = n - posicion
            candidatas = restantes[:m]
            puntuacion = mlp[actual, candidatas].astype(np.int64) * escala

            if intensificar:
                k = np.argmax(puntuacion - d[actual, candidatas])
            else:
                k = np.argmin(puntuacion + d[actual, candidatas])

            actual = int(candidatas[k])
            solucion[posicion] = actual
            restantes[k] = restantes[m - 1]

        return solucion

    def busqueda_tabu(self, seed: int):
        params = self.data.params

        max_iter = int(params.get("tabu_iter", 10000))
        tenencia = int(params.get("tabu_tenencia", 10))
        estancamiento = float(params.get("tabu_estancamiento", 0.05))
        oscilacion = float(params.get("tabu_oscilacion", 0.50))

        if max_iter <= 0 or tenencia <= 0:
            raise ValueError("Las iteraciones y la tenencia deben ser positivas")
        if not 0 < estancamiento <= 1:
            raise ValueError("El estancamiento debe estar en (0, 1]")
        if not 0 <= oscilacion <= 1:
            raise ValueError("La oscilación debe estar en [0, 1]")

        d = self.data.dist
        n = len(d)

        if n == 0:
            raise ValueError("El problema debe contener al menos una ciudad")

        limite_estancamiento = max(1, int(np.ceil(max_iter * estancamiento)))

        rng = np.random.default_rng(seed)

        tour, coste_actual = self.tsp_random_greedy(seed)
        s = np.asarray(tour, dtype=np.intp)
        coste_inicial = coste_actual

        mejor_global = s.copy()
        coste_mejor_global = coste_actual
        coste_mejor_momento = coste_actual

        it = 0

        # Con n <= 3 todos los tours cuestan lo mismo
        if n >= 4:
            # delta[p, q]: variación del coste al intercambiar las posiciones p y q.
            # fila_min[p] >= 0 indica que no hay mejora desde p (DLB exacto).
            delta = np.empty((n, n), dtype=np.int32)
            ant, sig, arcos, fila_min, fila_arg = self._construir_deltas(s, delta)

            pos = np.empty(n, dtype=np.intp)
            pos[s] = np.arange(n)

            # MCP: pares de ciudades intercambiadas en los últimos 'tenencia' movimientos
            lista_tabu = deque(maxlen=tenencia)

            # MLP simétrica con la frecuencia de cada arco.
            mlp = np.zeros((n, n), dtype=np.int32)
            entrada = np.zeros(n, dtype=np.int64)
            instantanea = 0
            escala = None

            en_n = np.zeros(n, dtype=bool)
            inicio = int(rng.integers(n))
            sin_mejora = 0

            while it < max_iter:
                # Posiciones vetadas para cada posición por la MCP
                vetadas = {}
                for c1, c2 in lista_tabu:
                    p1, p2 = int(pos[c1]), int(pos[c2])
                    vetadas.setdefault(p1, []).append(p2)
                    vetadas.setdefault(p2, []).append(p1)

                # Mínimo no tabú de cada fila: solo cambia si el mínimo es tabú
                efectivo = fila_min
                if vetadas:
                    efectivo = fila_min.copy()
                    for p, qs in vetadas.items():
                        if fila_arg[p] in qs:
                            fila = delta[p].copy()
                            fila[qs] = INT32_MAX
                            efectivo[p] = fila.min()

                filas = np.flatnonzero(efectivo < 0)
                mejora = filas.size > 0

                if not mejora:
                    # Mejor de los peores
                    objetivo = efectivo.min()
                    filas = np.flatnonzero(efectivo == objetivo)

                if mejora or objetivo != INT32_MAX:
                    # Primera fila en orden circular desde inicio y, en ella,
                    # primera columna en orden circular desde i + 1
                    i = int(filas[np.searchsorted(filas, inicio) % filas.size])

                    if mejora:
                        cols = np.flatnonzero(delta[i] < 0)
                    else:
                        cols = np.flatnonzero(delta[i] == objetivo)
                    if i in vetadas:
                        cols = cols[~np.isin(cols, vetadas[i])]

                    j = int(cols[np.searchsorted(cols, i + 1) % cols.size])

                    if mejora:
                        inicio = i

                    a, b = int(s[i]), int(s[j])

                    # MLP: se acumulan las instantáneas de los arcos que desaparecen
                    instantanea += 1
                    e = np.unique(np.array([i - 1, i, j - 1, j]) % n)
                    u, v = s[e], s[(e + 1) % n]
                    w = instantanea - entrada[e]
                    mlp[u, v] += w
                    mlp[v, u] += w
                    entrada[e] = instantanea

                    coste_actual += int(delta[i, j])
                    s[i], s[j] = b, a
                    pos[a], pos[b] = j, i

                    # Solo cambian las filas y columnas de las posiciones con
                    # algún arco nuevo
                    N = np.unique(np.array([i - 1, i, i + 1, j - 1, j, j + 1]) % n)
                    ant[N] = s[(N - 1) % n]
                    sig[N] = s[(N + 1) % n]
                    arcos[N] = d[ant[N], s[N]] + d[s[N], sig[N]]

                    nuevas = self._deltas_filas(s, ant, sig, arcos, N)
                    delta[N] = nuevas
                    delta[:, N] = nuevas.T

                    # Mínimos por fila: si baja, el nuevo mínimo está en N; si
                    # el antiguo mínimo estaba en N y no baja, se recalcula
                    cand_arg = nuevas.argmin(axis=0)
                    cand_min = nuevas[cand_arg, np.arange(n)]
                    baja = cand_min < fila_min
                    fila_min[baja] = cand_min[baja]
                    fila_arg[baja] = N[cand_arg[baja]]

                    en_n[N] = True
                    obsoletas = np.flatnonzero(en_n[fila_arg] & ~baja)
                    en_n[N] = False

                    obsoletas = np.union1d(obsoletas, N)
                    fila_arg[obsoletas] = delta[obsoletas].argmin(axis=1)
                    fila_min[obsoletas] = delta[obsoletas, fila_arg[obsoletas]]

                    lista_tabu.append((min(a, b), max(a, b)))

                    if coste_actual < coste_mejor_global:
                        mejor_global = s.copy()
                        coste_mejor_global = coste_actual

                    if coste_actual < coste_mejor_momento:
                        coste_mejor_momento = coste_actual
                        sin_mejora = 0
                    else:
                        sin_mejora += 1
                else:
                    # Todos los movimientos son tabú: se cuenta la iteración
                    # para no quedarse en un bucle infinito
                    sin_mejora += 1

                it += 1

                if sin_mejora >= limite_estancamiento and it < max_iter:
                    sin_mejora = 0
                    intensificar = bool(rng.random() < oscilacion)

                    # Volcado de la MLP antes de leerla
                    w = instantanea + 1 - entrada
                    mlp[s, sig] += w
                    mlp[sig, s] += w

                    if escala is None:
                        escala = int(d.max()) + 1

                    s = self._tabu_oscilacion(mlp, rng, intensificar, escala)
                    entrada.fill(instantanea + 1)

                    coste_actual = int(np.sum(d[s, np.roll(s, -1)], dtype=np.int64))
                    coste_mejor_momento = coste_actual

                    if coste_actual < coste_mejor_global:
                        mejor_global = s.copy()
                        coste_mejor_global = coste_actual

                    ant, sig, arcos, fila_min, fila_arg = self._construir_deltas(
                        s, delta
                    )
                    pos[s] = np.arange(n)
                    lista_tabu.clear()
                    inicio = int(rng.integers(n))

                    it += 1

        self.log_to_file(
            "tabu",
            coste_mejor_global,
            {
                "start": coste_inicial,
                "seed": seed,
                "tabu_iter": max_iter,
                "tabu_tenencia": tenencia,
                "tabu_estancamiento": estancamiento,
                "limite_estancamiento": limite_estancamiento,
                "tabu_oscilacion": oscilacion,
            },
            it,
            seed,
        )

        return mejor_global.tolist(), int(coste_mejor_global)

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
                    res = self.busqueda_tabu(seed)
                else:
                    pass

                t1 = time.perf_counter()
                elapsed = t1 - t0

                if res is not None:
                    tour, cost = res
                    costs.append(cost)
                    times.append(elapsed)
                    print(
                        f"[INFO] Algoritmo: {alg} | Seed: {seed} | Coste: {cost} | Tiempo: {elapsed:.4f}s"
                    )

            if costs:
                print(
                    f"[INFO] Resumen {alg} | Min: {min(costs)} | Media: {np.mean(costs):<10.2f} | Desv: {np.std(costs):<8.2f} | Tiempo medio: {np.mean(times):.4f}s"
                )
            else:
                print(f"[WARN] Algoritmo '{alg}' no implementado.")
