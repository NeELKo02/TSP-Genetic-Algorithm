import numpy as np
import random
import time
from functools import lru_cache
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

@lru_cache(maxsize=None)
def euc_dist(city1: tuple, city2: tuple) -> float:
    return np.sqrt((city1[0] - city2[0]) ** 2 + (city1[1] - city2[1]) ** 2 + (city1[2] - city2[2]) ** 2)

def calculate_path_dist(path: list) -> float:
    return sum(euc_dist(path[i], path[i + 1]) for i in range(len(path) - 1))

def knn(cities: list) -> list:
    start_city = random.choice(cities)
    path = [start_city]
    remaining_cities = set(cities) - {start_city}
    
    while remaining_cities:
        nearest_city = min(remaining_cities, key=lambda city: euc_dist(path[-1], city))
        path.append(nearest_city)
        remaining_cities.remove(nearest_city)
    
    path.append(path[0])
    return path

def get_pop_size(num_cities: int) -> int:
    return min(500, max(50, num_cities * 2))

def dynamic_initialization(cities: list, pop_size: int) -> list:
    num_cities = len(cities)
    knn_ratio = min(0.8, 1 - (num_cities / 1000))
    knn_count = int(pop_size * knn_ratio)

    population = [knn(cities) for _ in range(knn_count)]
    random_count = pop_size - knn_count
    population += [random.sample(cities, len(cities)) + [cities[0]] for _ in range(random_count)]
    
    return population

def create_mating_pool(population: list, fitness_scores: list) -> list:
    ranked_selection = sorted(zip(population, fitness_scores), key=lambda x: x[1], reverse=True)
    total_fitness = sum(fitness_scores)
    
    mating_pool = []
    for _ in range(len(population)):
        pick = random.uniform(0, total_fitness)
        current = 0
        for individual, score in ranked_selection:
            current += score
            if current > pick:
                mating_pool.append(individual)
                break
    
    return mating_pool

def two_pt_crossover(parent1: list, parent2: list) -> list:
    size = len(parent1) - 1
    start, end = sorted(random.sample(range(1, size), 2))
    child = [None] * size
    child[start:end] = parent1[start:end]
    
    remaining_cities = [city for city in parent2 if city not in child]
    index = 0
    for i in range(size):
        if child[i] is None:
            child[i] = remaining_cities[index]
            index += 1
    
    child.append(child[0])
    return child

def swap_mutation(path: list) -> None:
    size = len(path) - 1
    a, b = random.sample(range(1, size), 2)
    path[a], path[b] = path[b], path[a]

def inversion_mutation(path: list) -> None:
    size = len(path) - 1
    a, b = sorted(random.sample(range(1, size), 2))
    path[a:b] = list(reversed(path[a:b]))

def mutate(path: list, mut_rate: float) -> None:
    if random.random() < mut_rate:
        swap_mutation(path)
    if random.random() < mut_rate:
        inversion_mutation(path)

def gen_algo(cities: list, pop_size: int, generations: int, mut_rate: float, early_stop_threshold: int, elitism_fraction: float) -> tuple:
    start_time = time.time()
    population = dynamic_initialization(cities, pop_size)
    best_sol = None
    best_dist = float('inf')
    no_improvement_count = 0
    elitism_count = max(1, int(pop_size * elitism_fraction))

    for generation in range(generations):
        if time.time() - start_time > 290:
            logging.info("Stopping due to time limit.")
            break
        
        fitness_scores = [1 / calculate_path_dist(path) for path in population]
        next_population = []
        
        elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:elitism_count]
        for idx in elite_indices:
            next_population.append(population[idx])
        
        mating_pool = create_mating_pool(population, fitness_scores)
        while len(next_population) < pop_size:
            parent1, parent2 = random.sample(mating_pool, 2)
            child = two_pt_crossover(parent1, parent2)
            mutate(child, mut_rate)
            next_population.append(child)
        
        population = next_population
        current_best = min(population, key=calculate_path_dist)
        current_best_dist = calculate_path_dist(current_best)

        if current_best_dist < best_dist:
            best_sol = current_best
            best_dist = current_best_dist
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        if generation % 10 == 0:
            logging.info(f"Generation {generation}: best distance = {best_dist:.3f}")

        if no_improvement_count >= early_stop_threshold:
            logging.info(f"Stopping early at generation {generation} due to no improvement.")
            break

    return best_sol, best_dist

def read_input_file(input_file: str) -> list:
    with open(input_file, 'r') as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    cities = [tuple(map(int, line.strip().split())) for line in lines[1:n + 1]]
    return cities

def write_output_file(output_file: str, path: list, total_distance: float) -> None:
    with open(output_file, 'w') as f:
        f.write(f"{total_distance:.3f}\n")
        for city in path:
            f.write(f"{city[0]} {city[1]} {city[2]}\n")

def main():
    input_file = "input.txt"
    output_file = "output.txt"
    
    cities = read_input_file(input_file)
    pop_size = get_pop_size(len(cities))
    
    best_path, best_dist = gen_algo(
        cities, pop_size, generations=500, mut_rate=0.2,
        early_stop_threshold=50, elitism_fraction=0.1
    )

    write_output_file(output_file, best_path, best_dist)
    logging.info(f"Best tour distance: {best_dist:.3f}")

if __name__ == "__main__":
    main()
