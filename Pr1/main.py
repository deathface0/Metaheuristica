from tsp_loader import TSPLoader
from tsp_solver import TSPSolver

if __name__ == "__main__":
    loader = TSPLoader("preset.json")
    loader.load()

    for i in range(loader.length()):
        p = loader.parse()
        print(f"\n{'='*75}")
        print(f"[INFO] Instancia: {p.name} | Nodos: {p.dimension} | Tipo: {p.edge_weight_type}")
        print(f"{'='*75}")
        p.calc_dist()

        solver = TSPSolver(p)
        solver.run()

    
