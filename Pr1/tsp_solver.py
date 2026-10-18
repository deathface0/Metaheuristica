import json
import time

import numpy as np


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
            k = min(k, len(ordered_nodes))
            rand_idx = np.random.randint(0, k)
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
                    coste += delta[j]
                    dlb[[i, j]] = False
                    it += 1
                else:
                    dlb[i] = True
            self.log_to_file(
                "bl",
                coste,
                {"start": start, "bl_iter": max_iter, "seed": seed},
                int(it),
                seed,
            )
            i = (i + 1) % n

        return s.tolist(), int(coste)

    def _tabu_oscilacion(self, memoria, rng, intensificar):
        """Construye una solución usando las frecuencias de la MLP."""
        d = self.data.dist
        n = len(d)

        solucion = np.empty(n, dtype=np.intp)
        visitadas = np.zeros(n, dtype=bool)

        actual = int(rng.integers(n))
        solucion[0] = actual
        visitadas[actual] = True

        for posicion in range(1, n):
            candidatas = np.flatnonzero(~visitadas)

            # TSP simétrico: los arcos se almacenan en el triángulo inferior.
            frecuencias = memoria["mlp"][
                np.maximum(actual, candidatas),
                np.minimum(actual, candidatas),
            ]

            if intensificar:
                objetivo = frecuencias.max()
            else:
                objetivo = frecuencias.min()

            candidatas = candidatas[frecuencias == objetivo]

            # Desempate por distancia y, después, aleatoriamente.
            distancias = d[actual, candidatas]
            candidatas = candidatas[distancias == distancias.min()]

            actual = int(rng.choice(candidatas))
            solucion[posicion] = actual
            visitadas[actual] = True

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
        if not np.array_equal(d, d.T):
            raise ValueError("Esta implementación requiere un TSP simétrico")

        limite_estancamiento = max(1, int(np.ceil(max_iter * estancamiento)))

        rng = np.random.default_rng(seed)

        tour, coste_actual = self.tsp_random_greedy(seed)
        solucion_actual = np.asarray(tour, dtype=np.intp)
        coste_inicial = coste_actual

        mejor_momento_actual = solucion_actual.copy()
        coste_mejor_momento = coste_actual

        mejor_global = solucion_actual.copy()
        coste_mejor_global = coste_actual

        # Una única matriz cuadrada con celdas estructuradas:
        #
        # - MCP: triángulo superior, incluida la diagonal.
        #   Dos orientaciones para distinguir (ciudad, posición)
        #   de (posición, ciudad).
        #
        # - MLP: triángulo inferior, con frecuencias de arcos
        #   no dirigidos. Para n == 1 se utiliza su diagonal.
        memoria = np.zeros(
            (n, n),
            dtype=[
                ("mcp", np.int64, (2,)),
                ("mlp", np.int64),
            ],
        )

        mcp = memoria["mcp"]
        mlp = memoria["mlp"]

        # Solo se recorren las entradas activas al decrementar la MCP.
        activas = set()

        def clave_mcp(ciudad, posicion):
            return (
                min(ciudad, posicion),
                max(ciudad, posicion),
                int(ciudad > posicion),
            )

        def actualizar_mlp(solucion):
            siguientes = np.roll(solucion, -1)

            filas = np.maximum(solucion, siguientes)
            columnas = np.minimum(solucion, siguientes)

            # arco de vuelta al inicio tmb
            np.add.at(mlp, (filas, columnas), 1)

        actualizar_mlp(solucion_actual)
        sin_mejora = 0

        for it in range(max_iter):
            mejor_delta = np.inf
            movimiento = None

            for i in range(n - 1):
                # intercambiables con i (vecinos)
                js = np.arange(i + 1, n)

                a = int(solucion_actual[i])
                bs = solucion_actual[js]

                # mascaras para ver si intercambiar las 2 ciudades incumple el tabu
                tabu_a = (
                    mcp[
                        np.minimum(a, js),
                        np.maximum(a, js),
                        (a > js).astype(np.intp),
                    ]
                    > 0
                )

                tabu_b = (
                    mcp[
                        np.minimum(bs, i),
                        np.maximum(bs, i),
                        (bs > i).astype(np.intp),
                    ]
                    > 0
                )

                # intercambios admisibles dentro del vecindario, ya filtrados por la memoria tabu
                admisibles = js[~(tabu_a | tabu_b)]

                # duda
                if not admisibles.size:
                    continue

                # delta del coste como en bl
                deltas = self._deltas_intercambio(solucion_actual, i)

                # pillar negativo (mejora)
                mejoras = admisibles[deltas[admisibles] < 0]

                if mejoras.size:
                    j = int(mejoras[0])
                    movimiento = (i, j)
                    mejor_delta = int(deltas[j])
                    break

                # si no hay mejoras, delta mas pequeno
                j = int(admisibles[np.argmin(deltas[admisibles])])

                if deltas[j] < mejor_delta:
                    movimiento = (i, j)
                    mejor_delta = int(deltas[j])

            # A partir de aqui no se si esta bien, actualizar mcp, aplicar el moviemiento, actualizar mlp, actualizar solucion global, y mirar estancamiento

            # Actualización de MCP en CADA iteración.
            #
            # Se consulta antes de decrementar para que una entrada
            # con tenencia t bloquee las t selecciones siguientes.
            for clave in tuple(activas):
                mcp[clave] -= 1

                if mcp[clave] == 0:
                    activas.remove(clave)

            if movimiento is not None:
                i, j = movimiento
                a = int(solucion_actual[i])
                b = int(solucion_actual[j])

                # Prohibir que las ciudades vuelvan a sus posiciones
                # anteriores: dos asignaciones tabú por intercambio.
                for ciudad, posicion in ((a, i), (b, j)):
                    clave = clave_mcp(ciudad, posicion)
                    mcp[clave] = tenencia
                    activas.add(clave)

                solucion_actual[[i, j]] = solucion_actual[[j, i]]
                coste_actual += mejor_delta

            # Si no hay vecino admisible, se conserva la solución:
            # no se viola la MCP ni se aplica aspiración.

            # Actualización de MLP en CADA iteración.
            actualizar_mlp(solucion_actual)

            if coste_actual < coste_mejor_momento:
                mejor_momento_actual = solucion_actual.copy()
                coste_mejor_momento = coste_actual
                sin_mejora = 0
            else:
                sin_mejora += 1

            if coste_actual < coste_mejor_global:
                mejor_global = solucion_actual.copy()
                coste_mejor_global = coste_actual

            if sin_mejora >= limite_estancamiento and it + 1 < max_iter:
                # 50 % intensificación / 50 % diversificación.
                intensificar = bool(rng.random() < oscilacion)

                solucion_actual = self._tabu_oscilacion(memoria, rng, intensificar)

                coste_actual = int(
                    np.sum(
                        d[
                            solucion_actual,
                            np.roll(solucion_actual, -1),
                        ]
                    )
                )

                # Reiniciar el tramo, conservando MLP y mejor global.
                mcp.fill(0)
                activas.clear()
                sin_mejora = 0

                mejor_momento_actual = solucion_actual.copy()
                coste_mejor_momento = coste_actual

                # La nueva solución generada también ha sido visitada.
                actualizar_mlp(solucion_actual)

                if coste_actual < coste_mejor_global:
                    mejor_global = solucion_actual.copy()
                    coste_mejor_global = coste_actual

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
            max_iter,
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
                    pass
                elif alg == "gra":
                    pass
                elif alg == "bl":
                    pass
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
