from tsp_loader import TSPLoader
from tsp_solver import TSPSolver

if __name__ == "__main__":
    loader = TSPLoader("preset.json")
    loader.load()

    for i in range(loader.length()):
        p = loader.parse()
        print(f"\nCargando: {p.name} (ciudades: {p.dimension})")
        p.calc_dist()

        solver = TSPSolver(p)
        solver.run()

    
