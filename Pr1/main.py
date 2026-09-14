from tsp_loader import TSPLoader

if __name__ == "__main__":
    loader = TSPLoader("problems")
    loader.load()

    for i in range(5):
        p = loader.parse()
        print(f"size={p.datapoints.size} {p.name}: [{p.datapoints[-1]}]")
