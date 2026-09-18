import json
import os
from tsp_loader import TSPLoader

if __name__ == "__main__":
    with open("preset.json", "r", encoding="utf-8") as f:
        parametros = json.load(f)

    print("Parámetros cargados:", parametros)

    archivos_path = parametros["archivos"]
    if isinstance(archivos_path, list):
        archivos_path = archivos_path[0]

    loader = TSPLoader(archivos_path)
    loader.load()

    for i in range(loader.length()):
        p = loader.parse()
        print(f"\nCargado: {p.name} (ciudades: {p.dimension})")

        p.calc_dist()
        print(f"  Matriz de distancias calculada: {p.dist.shape}")

    
