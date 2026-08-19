# The Evolutionary Algorithm engine (thesis Section 3.6, "Genetic
# Algorithms") used during parameter tuning (tuning_pipeline.py, Section
# 4.3): population init, elitism, the two selection schemes, segment
# crossover, mutation, and the generation loop. The same algorithm is
# re-implemented in algorithms.py's run_ea for the actual Experiment 2 runs.

import random
import copy
import bisect
import numpy as np

# Random valuations used only while tuning (unnormalized, Uniform(1,1000))
# -- Experiment 1/2's own generators live in algorithms.py instead.
def generate_utilities(n, m):
    return [[random.uniform(1, 1000) for _ in range(m)] for _ in range(n)]


# Draws the initial population of pop_size random genes (Initialization
# paragraph, Section 3.6).
def initialize_population(pop_size, n, m):
    return [[random.randint(0, n - 1) for _ in range(m)] for _ in range(pop_size)]


# Obj_1-style EFX-violation count for one ordered (envious, envied) agent
# pair (Section 3.1, "Classification of Algorithms").
def count_violations_pair(envious_idx, envied_idx, bundles, utility_matrix, agent_utilities):
    violations = 0
    utility_of_envied_bundle = 0
    for item in bundles[envied_idx]:
        utility_of_envied_bundle += utility_matrix[envious_idx][item]
    for item in bundles[envied_idx]:
        if utility_of_envied_bundle - utility_matrix[envious_idx][item] > agent_utilities[envious_idx]:
            violations += 1
    return violations


# Obj_1 (total EFX-violation count) for one gene -- the fitness measure the
# GA minimizes.
def count_violations(n, m, solution, utility_matrix):
    bundles = [set() for _ in range(n)]
    violations = 0
    agent_utilities = [0.0] * n
    for item in range(m):
        bundles[solution[item]].add(item)
        agent_utilities[solution[item]] += utility_matrix[solution[item]][item]
    for i in range(n):
        for j in range(n):
            if i != j:
                violations += count_violations_pair(i, j, bundles, utility_matrix, agent_utilities)
    return violations


# ComputeBest from the Evolutionary Algorithm pseudocode (Section 3.6): the
# population's lowest-Obj_1 individual.
def compute_best(n, m, population, utility_matrix):
    population_violations = [0.0] * len(population)
    for i in range(len(population)):
        population_violations[i] = count_violations(n, m, population[i], utility_matrix)

    best_solution_idx = np.argmin(population_violations)
    best_solution = population[best_solution_idx]
    best = population_violations[best_solution_idx]

    return best_solution, best, population_violations


# Elitism: copies the e best individuals unchanged into the next generation
# (Selection paragraph, Section 3.6).
def elitism(population, elite_size, pop_violations):
    selected_pop = []
    sorted_indices = np.argsort(pop_violations)
    for i in range(elite_size):
        selected_pop.append(copy.deepcopy(population[sorted_indices[i]]))
    return selected_pop


# "Tournament" selection scheme (Selection paragraph): elitism plus
# repeated binary tournaments to fill the rest of the population.
def selection_with_elite(population, pop_size, elite_size, pop_violations):
    selected_pop = elitism(population, elite_size, pop_violations)
    while len(selected_pop) < pop_size:
        idx1, idx2 = random.sample(range(len(population)), 2)
        if pop_violations[idx1] <= pop_violations[idx2]:
            winner = population[idx1]
        else:
            winner = population[idx2]
        selected_pop.append(copy.deepcopy(winner))
    return selected_pop


# "Fitness-proportionate" selection scheme (Selection paragraph): samples
# proportionally to f(X) = worst Obj_1 in the population - Obj_1(X) + 1.
def selection_fitness_proportionate(population, pop_size, pop_violations):
    selected_pop = []
    max_violations = max(pop_violations)
    fitness = [(max_violations - pop_violations[i] + 1) for i in range(len(population))]
    total_fitness = sum(fitness)
    probabilities = [fitness[i] / total_fitness for i in range(len(population))]
    cum_probabilities = [probabilities[0]]
    for i in range(1, len(population)):
        cum_probabilities.append(copy.deepcopy(cum_probabilities[i - 1]) + probabilities[i])

    while len(selected_pop) < pop_size:
        r = random.random()
        idx = bisect.bisect_right(cum_probabilities, r)
        idx = min(idx, len(population) - 1)
        selected_pop.append(copy.deepcopy(population[idx]))

    return selected_pop


# Segment-based crossover (Crossover paragraph, Section 3.6): with
# probability cross_rate, each paired-up pair of genes swaps one
# contiguous, cyclic segment (proved fair in "Fairness of the Crossover
# Operator").
def crossover(n, m, population, pop_size, cross_rate):
    crossed_population = []
    population = list(population)
    random.shuffle(population)
    while len(crossed_population) < pop_size:
        if len(population) == 1:
            crossed_population.append(population[0])
            break

        parent1 = population.pop()
        parent2 = population.pop()

        if random.random() < cross_rate:
            child1 = copy.deepcopy(parent1)
            child2 = copy.deepcopy(parent2)

            size = random.randint(1, m - 1)
            start = random.randint(0, m - 1)
            end = (start + size) % m

            if start < end:
                child1[start:end] = parent2[start:end]
                child2[start:end] = parent1[start:end]
            else:
                child1[start:m] = parent2[start:m]
                child2[start:m] = parent1[start:m]
                child1[0:end] = parent2[0:end]
                child2[0:end] = parent1[0:end]

            crossed_population.append(child1)
            crossed_population.append(child2)
        else:
            crossed_population.append(copy.deepcopy(parent1))
            crossed_population.append(copy.deepcopy(parent2))

    return crossed_population


# Mutation (Mutation paragraph): independently for every gene and every
# good, reassigns it to a uniformly random agent with probability mut_rate.
def mutate(n, m, population, mut_rate):
    for sol_idx in range(len(population)):
        for item_idx in range(m):
            if random.random() < mut_rate:
                population[sol_idx][item_idx] = random.randint(0, n - 1)
    return population

# Dispatches to whichever of the two selection schemes this run uses.
def select_population(scheme, population, pop_size, elite_size, pop_violations):
    if scheme == "tournament":
        return selection_with_elite(population, pop_size, elite_size, pop_violations)
    elif scheme == "fitness_prop":
        elite = elitism(population, elite_size, pop_violations)
        rest = selection_fitness_proportionate(population, pop_size - elite_size, pop_violations)
        return elite + rest
    else:
        raise ValueError(f"Unknown selection scheme: {scheme!r}")

# The Evolutionary Algorithm's generation loop (Algorithm "Evolutionary
# Algorithm", Section 3.6): Select -> Crossover -> Mutate each generation
# until Obj_1=0 or max_gens is reached. Elitism keeps the best individual
# seen so far inside the population, so recomputing compute_best on the
# latest population each generation is equivalent to tracking the best ever.
def run_ea(n, m, pop_size, elite_size, cross_rate, mut_rate,
           selection_scheme, utility_matrix, max_gens):
    population = initialize_population(pop_size, n, m)
    best_solution, best, pop_violations = compute_best(n, m, population, utility_matrix)
    num_gens = 0
 
    while best != 0 and num_gens < max_gens:
        selected_pop = select_population(selection_scheme, population, pop_size, elite_size, pop_violations)
        crossed_pop = crossover(n, m, selected_pop, pop_size, cross_rate)
        population = mutate(n, m, crossed_pop, mut_rate)
        num_gens += 1
        best_solution, best, pop_violations = compute_best(n, m, population, utility_matrix)
 
    return {
        "best": best,
        "best_solution": best_solution,
        "num_gens": num_gens,
        "success": best == 0,
    }