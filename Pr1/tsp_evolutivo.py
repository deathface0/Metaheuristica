import time
from pathlib import Path

import numpy as np
from tsp_solver import TSPSolver


class TSPEvolutivo(TSPSolver):
    def _evaluar(self, solucion):
        d = self.data.dist
        return int(np.sum(d[solucion, np.roll(solucion, -1)]))

    def _torneo_mejor(self, costes, k_best, rng):
        candidatos = rng.choice(len(costes), size=k_best, replace=False)
        return int(candidatos[np.argmin(costes[candidatos])])

    def _torneo_peor(self, costes, disponibles, k_worst, rng):
        candidatos = rng.choice(disponibles, size=k_worst, replace=False)
        return int(candidatos[np.argmax(costes[candidatos])])

    def _hijo_ox2(self, donante, receptor, mascara):
        hijo = receptor.copy()
        valores = donante[mascara]
        posiciones = np.flatnonzero(np.isin(receptor, valores))
        hijo[posiciones] = valores
        return hijo

    def _cruce_ox2(self, padre1, padre2, rng):
        mascara = rng.random(len(padre1)) < 0.5
        hijo1 = self._hijo_ox2(padre1, padre2, mascara)
        hijo2 = self._hijo_ox2(padre2, padre1, mascara)
        return hijo1, hijo2

    def _hijo_moc(self, donante, receptor, corte):
        conservar = np.isin(receptor, donante[:corte])
        hijo = receptor.copy()
        hijo[~conservar] = donante[corte:]
        return hijo

    def _cruce_moc(self, padre1, padre2, rng):
        if len(padre1) < 2:
            return padre1.copy(), padre2.copy()
        corte = int(rng.integers(1, len(padre1)))
        hijo1 = self._hijo_moc(padre1, padre2, corte)
        hijo2 = self._hijo_moc(padre2, padre1, corte)
        return hijo1, hijo2

    def _mutacion(self, individuo, rng):
        if len(individuo) >= 2 and rng.random() < 0.10:
            i, j = rng.choice(len(individuo), size=2, replace=False)
            individuo[[i, j]] = individuo[[j, i]]

    # poblacion = nuevos descendientes
    # costes
    # los E mejores de la poblacion anterior
    # costes
    # kWorst
    def _elitismo(self, poblacion, costes, elites, costes_elite, k_worst, rng):

        # representar indices de la poblacion como array de bools para marcar si elite o no
        protegidos = np.zeros(len(poblacion), dtype=bool)
        ausentes = []

        # recorrer individuos elites y su coste
        for elite, coste in zip(elites, costes_elite):
            # cada elemento de la poblacion con elite y comprobar que todos los elementos de la fila sean true
            # con np.all y mirar los individuos true, (true = son iguales a elite), todo entre
            # los valores que no estan protegidos
            iguales = np.flatnonzero(~protegidos & np.all(poblacion == elite, axis=1))
            if iguales.size:
                # proteger primera coincidencia
                protegidos[iguales[0]] = True
            else:
                ausentes.append((elite, coste))

        # todos los no protegidos participan en torneo perdedores
        for elite, coste in ausentes:
            disponibles = np.flatnonzero(~protegidos)
            # victima segun kworst
            peor = self._torneo_peor(costes, disponibles, k_worst, rng)
            poblacion[peor] = elite
            costes[peor] = coste
            protegidos[peor] = True

    def evolutivo_generacional(self, seed: int):
        inicio = time.perf_counter()
        params = self.data.params
        elite = int(params.get("gen_elite", 1))
        k_best = int(params.get("gen_k_best", 2))
        k_worst = int(params.get("gen_k_worst", 2))
        cruce = str(params.get("gen_cruce", "ox2")).lower()
        max_eval = int(params.get("gen_max_eval", 50000))
        max_seconds = float(params.get("gen_max_seconds", 120))

        if elite not in (1, 2, 3):
            raise ValueError("gen_elite debe ser 1, 2 o 3")
        if k_best not in (2, 3) or k_worst not in (2, 3):
            raise ValueError("Los tamaños de torneo deben ser 2 o 3")
        if cruce not in ("ox2", "moc"):
            raise ValueError("gen_cruce debe ser ox2 o moc")
        if max_eval < 100:
            raise ValueError("gen_max_eval debe ser al menos 100")
        if not np.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError("gen_max_seconds debe ser positivo y finito")
        if params.get("k") != 10:
            raise ValueError('Configura params["k"] = 10 para el greedy original')

        d = np.asarray(self.data.dist)
        n = len(d)
        if d.ndim != 2 or n == 0 or d.shape != (n, n):
            raise ValueError("Se requiere una matriz cuadrada no vacía")

        Path("logs").mkdir(exist_ok=True)
        rng = np.random.default_rng(seed)
        limite_tiempo = inicio + max_seconds
        evaluaciones = 0
        generaciones = 0
        mejor_global = None
        coste_mejor = None
        poblacion = []
        costes = []

        # 20 semillas para el 20% de individuos mediante gra
        semillas_greedy = rng.choice(2**31 - 1, size=20, replace=False)

        for i in range(100):
            if time.perf_counter() >= limite_tiempo:
                break
            # por debajo de 80, permutacion aleatoria, y greedy aleatorio para las 20 iter restantes
            if i < 80:
                solucion = rng.permutation(n)
                if time.perf_counter() >= limite_tiempo:
                    break
                coste = self._evaluar(solucion)
            else:
                tour, coste = self.tsp_random_greedy(int(semillas_greedy[i - 80]))
                solucion = np.asarray(tour, dtype=np.intp)
            evaluaciones += 1
            poblacion.append(solucion)
            costes.append(coste)
            if coste_mejor is None or coste < coste_mejor:
                mejor_global = solucion.copy()
                coste_mejor = coste

        if mejor_global is None:
            raise TimeoutError("Tiempo agotado antes de evaluar una solución")

        poblacion = np.asarray(poblacion, dtype=np.intp)
        costes = np.asarray(costes)
        coste_inicial = int(coste_mejor)

        # mirar que la poblacion este completa y que todavia quede tiempo
        while len(poblacion) == 100 and time.perf_counter() < limite_tiempo:
            # elitismo segun parametro, ordenamos los indices y obtenemos los E mejores
            indices = np.argsort(costes, kind="stable")[:elite]
            elites = poblacion[indices].copy()
            costes_elite = costes[indices].copy()

            # torneo mejor para escoger padres con kBEst igual al numero de individuos
            padres = []
            for _ in range(100):
                if time.perf_counter() >= limite_tiempo:
                    break
                padres.append(self._torneo_mejor(costes, k_best, rng))
            if len(padres) != 100:
                break

            # generacion de descendientes (nueva poblacion) mediante cruces
            descendientes = []
            costes_descendientes = []
            # padres de dos en dos
            for i in range(0, 100, 2):
                if time.perf_counter() >= limite_tiempo:
                    break
                padre1 = poblacion[padres[i]]
                padre2 = poblacion[padres[i + 1]]
                # 70% si se cruza la pareja o no
                if rng.random() < 0.70:
                    if cruce == "ox2":
                        hijos = self._cruce_ox2(padre1, padre2, rng)
                    else:
                        hijos = self._cruce_moc(padre1, padre2, rng)
                else:
                    hijos = padre1.copy(), padre2.copy()

                # evaluacion de hijos y mutacion
                for hijo in hijos:
                    if time.perf_counter() >= limite_tiempo:
                        break
                    self._mutacion(hijo, rng)
                    if time.perf_counter() >= limite_tiempo:
                        break
                    coste = self._evaluar(hijo)
                    evaluaciones += 1
                    descendientes.append(hijo)
                    costes_descendientes.append(coste)
                    if coste < coste_mejor:
                        mejor_global = hijo.copy()
                        coste_mejor = coste

            if len(descendientes) != 100:
                break
            descendientes = np.asarray(descendientes, dtype=np.intp)
            costes_descendientes = np.asarray(costes_descendientes)
            self._elitismo(
                descendientes, costes_descendientes, elites, costes_elite, k_worst, rng
            )
            poblacion = descendientes
            costes = costes_descendientes
            generaciones += 1

        self.log_to_file(
            f"gen_{cruce}_e{elite}_kb{k_best}_kw{k_worst}",
            int(coste_mejor),
            {
                "start": coste_inicial,
                "seed": seed,
                "k": params["k"],
                "gen_elite": elite,
                "gen_k_best": k_best,
                "gen_k_worst": k_worst,
                "gen_cruce": cruce,
                "gen_max_eval": max_eval,
                "gen_max_seconds": max_seconds,
                "evaluaciones": evaluaciones,
                "tiempo": time.perf_counter() - inicio,
                "inicializacion_completa": len(poblacion) == 100,
            },
            generaciones,
            seed,
        )
        return mejor_global.tolist(), int(coste_mejor)

    def run(self):
        for alg in self.data.algorithms:
            costs = []
            times = []
            for seed in self.data.seeds:
                t0 = time.perf_counter()
                if alg == "gen":
                    tour, cost = self.evolutivo_generacional(seed)
                else:
                    continue
                elapsed = time.perf_counter() - t0
                costs.append(cost)
                times.append(elapsed)
                print(
                    f"[INFO] Algoritmo: {alg} | Seed: {seed} | "
                    f"Coste: {cost} | Tiempo: {elapsed:.4f}s"
                )
            if costs:
                print(
                    f"[INFO] Resumen {alg} | Min: {min(costs)} | "
                    f"Media: {np.mean(costs):<10.2f} | "
                    f"Desv: {np.std(costs):<8.2f} | "
                    f"Tiempo medio: {np.mean(times):.4f}s"
                )
            else:
                print(f"[WARN] Algoritmo '{alg}' no implementado en este archivo.")
