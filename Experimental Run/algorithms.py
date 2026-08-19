"""
algorithms.py

Every algorithm compared in the thesis's Algorithms section (Section 3):
the 4 ILP formulations (Section 3.2), the Ratio-Greedy algorithm
(Section 3.3), Round-Robin (Section 3.4), Simulated Annealing with its
four warm starts (Section 3.5), and the Evolutionary Algorithm
(Section 3.6). Also included: the EFX/EF checks and the Obj_1..Obj_4
objective functions from Section 3.1's "Classification of Algorithms",
and the two instance generators used to build experiment instances --
i.i.d. uniform (Section 4.1) and the resampling model (Section 2.2).
"""

import random
import copy
import bisect
import math
import os
import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    gp = None
    GRB = None


def make_gurobi_env():
    """Builds the Gurobi environment used by the 4 ILP solvers below
    (solve_ilp_1..4), which implement the exact-benchmark ILP
    formulations of Section 3.2, "ILP".
    """
    if gp is None:
        raise ImportError("gurobipy is not installed -- pip install gurobipy")

    required = ["GRB_WLSACCESSID", "GRB_WLSSECRET", "GRB_LICENSEID"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            f"Missing WLS credentials in environment: {missing}. "
            f"Set them (e.g. in ~/.bashrc on Snellius) before creating a Gurobi Env."
        )

    return gp.Env(params={
        "WLSACCESSID": os.environ["GRB_WLSACCESSID"],
        "WLSSECRET": os.environ["GRB_WLSSECRET"],
        "LICENSEID": int(os.environ["GRB_LICENSEID"]),
    })

# Floating-point tolerance used throughout when checking EFX/envy inequalities
# (e.g. is_efx below), since valuations are real-valued.
EPS = 0.0000001


def bundles_to_allocation(bundles):
    """Converts the list-of-sets bundle representation used by Simulated
    Annealing (Section 3.5) and Round-Robin (Section 3.4) into the
    canonical per-agent list-of-lists allocation format.
    """
    return [sorted(b) for b in bundles]


def solution_to_allocation(n, solution):
    """Converts the Evolutionary Algorithm's gene representation -- one
    owning agent per good, see the "Solution Representation" paragraph
    of Section 3.6 -- into the canonical per-agent allocation format.
    """
    allocation = [[] for _ in range(n)]
    for item, agent in enumerate(solution):
        allocation[agent].append(item)
    return allocation



# Agent's valuation of a bundle, v_i(bundle), as defined in the thesis's
# Fair Division Preliminaries section.
def compute_utility(agent, utility_matrix, bundle):
    utility = 0
    for good in bundle:
        utility += utility_matrix[agent][good]
    return utility


# Checks whether an allocation is exactly EFX: no agent envies another agent's
# bundle even after removing that bundle's least-valuable good from the
# envious agent's own perspective (the EFX definition in Fair Division
# Preliminaries).
def is_efx(allocation, utility_matrix, n, m):
    own_utilities = [sum(utility_matrix[i][g] for g in allocation[i]) for i in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            val_of_j_bundle = sum(utility_matrix[i][g] for g in allocation[j])
            if own_utilities[i] >= val_of_j_bundle - EPS:
                continue
            if not allocation[j]:
                continue
            for g in allocation[j]:
                if own_utilities[i] < (val_of_j_bundle - utility_matrix[i][g] - EPS):
                    return False
    return True


# Checks whether an allocation is envy-free (EF): no agent values another
# agent's full bundle more than their own (the EF definition in Fair
# Division Preliminaries).
def is_envy_free(n, v, bundles, agent_utilities):
    for i in range(n):
        u_i = agent_utilities[i]
        for j in range(n):
            if i == j:
                continue
            u_ij = sum(v[i][k] for k in bundles[j])
            if u_ij > u_i + EPS:
                return False
    return True



def evaluate_all_objectives(allocation, v):
    """Computes the four EFX-distance objective functions Obj_1..Obj_4
    (Section 3.1, "Classification of Algorithms") in a single pass over
    the allocation. Obj_1 and Obj_2 sum violation count / envy
    magnitude over all agents (utilitarian); Obj_4 and Obj_3 instead
    take the worst single agent's count / magnitude (egalitarian).
    Returns (objective_1, objective_2, objective_3, objective_4).
    """
    n = len(allocation)
    agent_utilities = [sum(v[i][g] for g in allocation[i]) for i in range(n)]

    obj1_total = 0
    obj2_total = 0.0
    obj4_worst = 0
    obj3_worst = 0.0

    for i in range(n):
        count_i = 0
        magnitude_i = 0.0
        for j in range(n):
            if i == j:
                continue
            u_ij = sum(v[i][g] for g in allocation[j])
            for g in allocation[j]:
                envy_after_removing_g = (u_ij - agent_utilities[i]) - v[i][g]
                if envy_after_removing_g > EPS:
                    count_i += 1
                    magnitude_i += envy_after_removing_g
        obj1_total += count_i
        obj2_total += magnitude_i
        obj4_worst = max(obj4_worst, count_i)
        obj3_worst = max(obj3_worst, magnitude_i)

    return obj1_total, obj2_total, obj3_worst, obj4_worst


# Obj_1: total EFX-violation count (also the objective used by Simulated
# Annealing, see calculate_full_potential below).
def objective_1(allocation, v):
    return evaluate_all_objectives(allocation, v)[0]


# Obj_2: total EFX-envy magnitude, summed over all agents.
def objective_2(allocation, v):
    return evaluate_all_objectives(allocation, v)[1]


# Obj_3: egalitarian EFX-envy magnitude, i.e. the worst single agent's total envy.
def objective_3(allocation, v):
    return evaluate_all_objectives(allocation, v)[2]


# Obj_4: egalitarian EFX-violation count, i.e. the worst single agent's violation count.
def objective_4(allocation, v):
    return evaluate_all_objectives(allocation, v)[3]


# Counts Obj_1-style EFX violations for a single ordered (envious, envied)
# agent pair only -- the building block that lets find_efx_allocation below
# recompute Obj_1's change incrementally after moving just one good, instead
# of recomputing the full objective from scratch at every SA step.
def _count_violations_for_pair(envious_idx, envied_idx, v, bundles, agent_utilities):
    u_envious_own = agent_utilities[envious_idx]
    envied_bundle = bundles[envied_idx]
    u_envious_of_envied = sum(v[envious_idx][item] for item in envied_bundle)
    violations = 0
    for item_k in envied_bundle:
        if u_envious_of_envied - v[envious_idx][item_k] > u_envious_own + EPS:
            violations += 1
    return violations


# Full Obj_1 value (total EFX-violation count) for an allocation -- the
# objective function Simulated Annealing (below) searches over, matching
# the thesis's Simulated Annealing section.
def calculate_full_potential(n, v, bundles, agent_utilities):
    total_violations = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            total_violations += _count_violations_for_pair(i, j, v, bundles, agent_utilities)
    return total_violations


def initialize_allocation(n, m, v):
    """The "Random" warm start for Simulated Annealing (Section 3.5,
    "Warm Starts"): every good is assigned to a uniformly random
    agent.
    """
    bundles = [set() for _ in range(n)]
    owner = [-1] * m
    agent_utilities = [0.0] * n
    for item_j in range(m):
        agent_i = random.randint(0, n - 1)
        bundles[agent_i].add(item_j)
        owner[item_j] = agent_i
        agent_utilities[agent_i] += v[agent_i][item_j]
    return bundles, owner, agent_utilities


def initialize_allocation_from_optimal_welfare(n, m, v):
    """The "Welfare" warm start (Section 3.5, "Warm Starts"): assigns
    every good to whichever agent values it most, exactly maximizing
    the utilitarian welfare SW(X) of Section 2.1.
    """
    bundles = [set() for _ in range(n)]
    owner = [-1] * m
    agent_utilities = [0.0] * n
    for item_j in range(m):
        agent_i = -1
        highest_value_for_j = -1
        for i in range(n):
            if v[i][item_j] > highest_value_for_j:
                agent_i = i
                highest_value_for_j = v[i][item_j]
        bundles[agent_i].add(item_j)
        owner[item_j] = agent_i
        agent_utilities[agent_i] += v[agent_i][item_j]
    return bundles, owner, agent_utilities


def initialize_allocation_from_round_robin(n, m, v):
    """The "Round-Robin" warm start (Section 3.5, "Warm Starts"): starts
    Simulated Annealing from the Round-Robin allocation
    (round_robin_allocation below), which is EF1 by construction
    (Section 3.4).
    """
    return round_robin_allocation(n, m, v)


def initialize_allocation_from_greedy(n, m, v):
    """The "Greedy" warm start (Section 3.5, "Warm Starts"): starts
    Simulated Annealing from the Ratio-Greedy allocation
    (find_efx_allocation_greedy below, Section 3.3).
    """
    allocation, current_utilities = find_efx_allocation_greedy(n, m, v)
    bundles = [set(a) for a in allocation]
    owner = [-1] * m
    for i in range(n):
        for g in allocation[i]:
            owner[g] = i
    return bundles, owner, list(current_utilities)


def find_efx_allocation(n, m, v, bundles, owner, agent_utilities, max_steps=10_000_000):
    """Implements the thesis's Simulated Annealing algorithm for EFX
    (Algorithm "Simulated annealing for EFX", Section 3.5): proposes
    single-good-transfer neighbors (Section 3.5.1, "Neighborhood
    Structure") and accepts/rejects each proposal via the Metropolis
    rule on Obj_1, with T0=5.0, T_min=0.0001, cooling_rate=0.99 and
    100*n*m proposals per temperature level -- all unchanged from
    Branzei et al.

    Implements both deviations from Branzei et al. described in
    Section 3.5.3, "Deviations from Branzei et al.":

    1. No restarts. One cooling pass only -- it stops on success
       (best_f_value hits 0), the cooling schedule running out (T
       drops below T_min), or max_steps as a safety cap (Section 3.5,
       "Design": at most ~1077 temperature levels per pass, i.e. up
       to roughly 107,700 * n * m steps in the worst case).

    2. Always returns the best allocation ever seen, not just
       whatever the current state is when the function stops, since
       without restarts the search can now stop above 0 violations,
       and Simulated Annealing deliberately accepts some worse moves
       along the way.

    Returns (num_steps, allocation_bundles, owner, agent_utilities).
    allocation_bundles is never None -- check is_efx() on the result
    to see whether it actually found an EFX allocation, versus just
    its best effort.
    """
    num_steps = 0

    f = calculate_full_potential(n, v, bundles, agent_utilities)
    best_f_value = f
    best_bundles = copy.deepcopy(bundles)
    best_owner = list(owner)
    best_agent_utilities = list(agent_utilities)

    T = 5.0
    T_min = 0.0001
    cooling_rate = 0.99

    
    while T > T_min and best_f_value > 0:
        steps_per_temp = 100 * n * m

        for _ in range(steps_per_temp):
            if best_f_value == 0:
                break

            item_to_move = random.randint(0, m - 1)
            owner_old = owner[item_to_move]
            owner_new = random.randint(0, n - 1)
            while owner_new == owner_old:
                owner_new = random.randint(0, n - 1)

            affected_agents = {owner_old, owner_new}

            old_slice_f = 0
            for i in range(n):
                for j in affected_agents:
                    if i != j:
                        old_slice_f += _count_violations_for_pair(i, j, v, bundles, agent_utilities)
            for i in affected_agents:
                for j in range(n):
                    if i != j and j not in affected_agents:
                        old_slice_f += _count_violations_for_pair(i, j, v, bundles, agent_utilities)

            bundles[owner_old].remove(item_to_move)
            bundles[owner_new].add(item_to_move)
            agent_utilities[owner_old] -= v[owner_old][item_to_move]
            agent_utilities[owner_new] += v[owner_new][item_to_move]

            new_slice_f = 0
            for i in range(n):
                for j in affected_agents:
                    if i != j:
                        new_slice_f += _count_violations_for_pair(i, j, v, bundles, agent_utilities)
            for i in affected_agents:
                for j in range(n):
                    if i != j and j not in affected_agents:
                        new_slice_f += _count_violations_for_pair(i, j, v, bundles, agent_utilities)

            delta_f = new_slice_f - old_slice_f

            accept_move = False
            if delta_f < 0:
                accept_move = True
            else:
                if T > 0:
                    acceptance_probability = math.exp(-delta_f / T)
                    if random.uniform(0, 1) < acceptance_probability:
                        accept_move = True

            if accept_move:
                owner[item_to_move] = owner_new
                f += delta_f
            else:
                bundles[owner_new].remove(item_to_move)
                bundles[owner_old].add(item_to_move)
                agent_utilities[owner_new] -= v[owner_new][item_to_move]
                agent_utilities[owner_old] += v[owner_old][item_to_move]

            if f < best_f_value:
                best_f_value = f
                best_bundles = copy.deepcopy(bundles)
                best_owner = list(owner)
                best_agent_utilities = list(agent_utilities)

            num_steps += 1

            if num_steps >= max_steps:
                return num_steps, best_bundles, best_owner, best_agent_utilities

        if best_f_value == 0:
            break
        T *= cooling_rate

    return num_steps, best_bundles, best_owner, best_agent_utilities


# The Round-Robin allocation algorithm (Round-Robin section): agents take
# turns in a fixed cyclic order, each picking their most-valued remaining
# good; guarantees EF1 but not EFX.
def round_robin_allocation(n, m, v):
    bundles = [set() for _ in range(n)]
    owner = [-1] * m
    agent_utilities = [0.0] * n
    items_remaining = list(range(m))
    while len(items_remaining) > 0:
        for i in range(n):
            if len(items_remaining) == 0:
                break
            items_remaining.sort(key=lambda p: v[i][int(p)], reverse=True)
            j = items_remaining[0]
            bundles[i].add(j)
            owner[j] = i
            agent_utilities[i] += v[i][j]
            items_remaining.remove(j)
    return bundles, owner, agent_utilities



# Helper for the Ratio-Greedy algorithm (Ratio-Greedy Algorithm section):
# picks the remaining good with the highest valuation ratio for one agent.
def get_max_good(available_goods, agent, ratio_matrix):
    max_good = available_goods[0]
    for good in available_goods:
        if ratio_matrix[agent][good] > ratio_matrix[agent][max_good]:
            max_good = good
    return max_good


# Computes the valuation-ratio matrix R (Valuation ratio definition):
# R_{i,k} = v_{i,k} / sum_j v_{j,k}, how much agent i values good k relative
# to everyone else.
def compute_ratio_matrix(n, m, utility_matrix):
    utility_arr = np.asarray(utility_matrix)
    total_good_utility = np.sum(utility_arr, axis=0)
    ratio_matrix = [[0 for _ in range(m)] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            if utility_matrix[i][j] != 0 and total_good_utility[j] != 0:
                ratio_matrix[i][j] = utility_matrix[i][j] / total_good_utility[j]
    return ratio_matrix


# One round of the Ratio-Greedy algorithm: orders agents poorest-first by
# current utility and gives the poorest agent (skipping any for whom every
# remaining good has ratio 0) their highest-ratio remaining good.
def allocate_next_good(n, current_utilities, available_goods, ratio_matrix):
    agents_by_utility = np.argsort(current_utilities)
    for agent in agents_by_utility:
        max_available_good = get_max_good(available_goods, agent, ratio_matrix)
        if ratio_matrix[agent][max_available_good] > 0:
            return agent, max_available_good
    agent = agents_by_utility[0]
    max_available_good = get_max_good(available_goods, agent, ratio_matrix)
    return agent, max_available_good


# The Ratio-Greedy allocation algorithm (Algorithm "Ratio-greedy allocation",
# Ratio-Greedy Algorithm section): repeatedly allocates one good at a time to
# the poorest agent until every good is allocated. Runs in worst-case
# O(n*m^2) time; has no EFX or EF1 quality guarantee (proved by
# counterexample in the same section).
def find_efx_allocation_greedy(n, m, utility_matrix):
    ratio_matrix = compute_ratio_matrix(n, m, utility_matrix)
    current_utilities = [0 for _ in range(n)]
    available_goods = [g for g in range(m)]
    allocation = [[] for i in range(n)]
    while len(available_goods) > 0:
        agent, max_available_good = allocate_next_good(n, current_utilities, available_goods, ratio_matrix)
        available_goods.remove(max_available_good)
        allocation[agent].append(max_available_good)
        current_utilities[agent] += utility_matrix[agent][max_available_good]
    return allocation, current_utilities



# Evolutionary Algorithm (Genetic Algorithms section): draws the initial
# population of p solutions independently and uniformly at random
# (Initialization paragraph).
def initialize_population(pop_size, n, m):
    return [[random.randint(0, n - 1) for _ in range(m)] for _ in range(pop_size)]


# Same Obj_1-style pairwise violation count as _count_violations_for_pair
# above, operating on the GA's bundle representation.
def count_violations_pair(envious_idx, envied_idx, bundles, utility_matrix, agent_utilities):
    violations = 0
    utility_of_envied_bundle = 0
    for item in bundles[envied_idx]:
        utility_of_envied_bundle += utility_matrix[envious_idx][item]
    for item in bundles[envied_idx]:
        if utility_of_envied_bundle - utility_matrix[envious_idx][item] > agent_utilities[envious_idx] + EPS:
            violations += 1
    return violations


# Obj_1 (total EFX-violation count) for one GA individual/gene -- the
# fitness measure the Evolutionary Algorithm minimizes.
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


# ComputeBest from the Evolutionary Algorithm pseudocode: the population's
# lowest-Obj_1 individual.
def compute_best(n, m, population, utility_matrix):
    population_violations = [0.0] * len(population)
    for i in range(len(population)):
        population_violations[i] = count_violations(n, m, population[i], utility_matrix)
    best_solution_idx = np.argmin(population_violations)
    best_solution = population[best_solution_idx]
    best = population_violations[best_solution_idx]
    return best_solution, best, population_violations


# Elitism: copies the e best individuals unchanged into the next generation
# (Selection paragraph, Genetic Algorithms section).
def elitism(population, elite_size, pop_violations):
    selected_pop = []
    sorted_indices = np.argsort(pop_violations)
    for i in range(elite_size):
        selected_pop.append(copy.deepcopy(population[sorted_indices[i]]))
    return selected_pop


# "Tournament" selection scheme: elitism plus repeated binary tournaments
# (better of two random individuals) to fill the rest of the population
# (Selection paragraph).
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


# "Fitness-proportionate" selection scheme: converts Obj_1 into a positive
# fitness f(X) = worst Obj_1 in the population - Obj_1(X) + 1 and samples
# proportionally to it (Selection paragraph).
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


# Segment-based crossover (Crossover paragraph): with probability cross_rate,
# each paired-up pair of genes swaps one contiguous, cyclic segment; proved
# fair (every good equally likely to be swapped) in "Fairness of the
# Crossover Operator".
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


# Dispatches to whichever of the two selection schemes (Selection paragraph)
# this GA run was configured with.
def select_population(scheme, population, pop_size, elite_size, pop_violations):
    if scheme == "tournament":
        return selection_with_elite(population, pop_size, elite_size, pop_violations)
    elif scheme == "fitness_prop":
        elite = elitism(population, elite_size, pop_violations)
        rest = selection_fitness_proportionate(population, pop_size - elite_size, pop_violations)
        return elite + rest
    else:
        raise ValueError(f"Unknown selection scheme: {scheme!r}")


# The Evolutionary Algorithm (Algorithm "Evolutionary Algorithm", Genetic
# Algorithms section): evolves the population via Select -> Crossover ->
# Mutate each generation, tracking the best (lowest-Obj_1) solution seen,
# until an EFX solution (Obj_1=0) is found or max_gens is reached.
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
        gen_best_solution, gen_best, pop_violations = compute_best(n, m, population, utility_matrix)
        if gen_best < best:
            best, best_solution = gen_best, gen_best_solution
    return {
        "best": best,
        "best_solution": best_solution,
        "num_gens": num_gens,
        "success": best == 0,
    }


def generate_iid_uniform(n, m):
    """Experiment 1's valuation generator (Section 4.1, "Good-to-Agent
    Ratio and Warm Starts"): every v_{i,k} is drawn i.i.d. ~
    Uniform(0,1), matching Branzei et al.'s own generator.
    """
    return [[random.uniform(0, 1) for _ in range(m)] for _ in range(n)]


# The (p, phi) grid used to generate Experiment 2's instances, matching the
# Resampling Generator section (and Bohm et al.'s own grid).
P_VALUES = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
PHI_VALUES = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]


# Resampling Generator, step 1: draws the central approval set V* of size
# floor(p*m) uniformly at random.
def generate_central_aproval_set(m, p):
    goods = [i for i in range(m)]
    random.shuffle(goods)
    cas = goods[:math.floor(p * m)]
    return cas


# Resampling Generator, steps 2-3: each agent approves V* with probability
# 1-phi or an independent p-random set with probability phi, then any agent
# left with no approved goods is given one uniformly at random.
def generate_approval_matrix(n, m, p, phi):
    cas = generate_central_aproval_set(m, p)
    approved = [[0 for j in range(m)] for i in range(n)]
    for i in range(n):
        for j in range(m):
            if random.random() < 1 - phi:
                if j in cas:
                    approved[i][j] = 1
            else:
                if random.random() < p:
                    approved[i][j] = 1
        if sum(approved[i]) == 0:
            approved[i][random.randint(0, m - 1)] = 1
    return approved


def generate_utility_matrix(n, m, p, phi):
    """Resampling Generator (Section 2.2), step 4, for an explicit
    (p, phi): splits each agent's one unit of utility equally over
    their approved goods (built by generate_approval_matrix above).
    """
    approval_matrix = generate_approval_matrix(n, m, p, phi)
    total_approved = [sum(approval_matrix[i]) for i in range(n)]
    utility_matrix = [[approval_matrix[i][j] / total_approved[i] for j in range(m)] for i in range(n)]
    return utility_matrix


def generate_utilities_resampling(n, m):
    """Resampling Generator (Section 2.2) with (p, phi) drawn from the
    P_VALUES/PHI_VALUES grid above -- used to generate Experiment 2's
    instances (Section 4.2). Returns (utility_matrix, p, phi).
    """
    p = random.choice(P_VALUES)
    phi = random.choice(PHI_VALUES)
    return generate_utility_matrix(n, m, p, phi), p, phi


def explicit_map(utility_matrix):
    """The explicit map mu(v) = (sigma_1, sigma_2) of Section 2.3, "The
    Explicit Map and Characteristic Instances": the two largest
    singular values of the valuation matrix.
    """
    utility_matrix = np.asarray(utility_matrix, dtype=float)
    singular_values = np.linalg.svd(utility_matrix, compute_uv=False)
    sigma_1 = singular_values[0]
    sigma_2 = singular_values[1] if len(singular_values) > 1 else 0.0
    return sigma_1, sigma_2


# Reads a Gurobi solution's x_{i,k} variables back into the canonical
# per-agent allocation format.
def _extract_allocation(x, n, m):
    return [[g for g in range(m) if x[i, g].X > 0.5] for i in range(n)]


# Maps Gurobi's solve status onto the run's recorded "status" field.
def _solve_status(model):
    if model.status == GRB.OPTIMAL:
        return "optimal"
    if model.status == GRB.TIME_LIMIT:
        return "time_limit" if model.SolCount > 0 else "no_solution_found"
    return f"other:{model.status}"


def solve_ilp_1(n, m, v, M, env, time_limit=None, mip_gap=None):
    """ILP 1: Total EFX Violation Count (Section 3.2). Minimizes
    sum_{i,j,k} y_{i,j,k}, with decision variables x/w/y and
    constraints matching Section 3.2's eq:assignment/eq:b/eq:c/eq:d/eq:e.
    """
    model = gp.Model("Minimizing EFX Violations", env=env)
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap

    # Decision variables (ILP section): x_{i,k}, w_{i,j,k}, y_{i,j,k}.
    x = model.addVars(n, m, vtype=GRB.BINARY, name='x')
    w = model.addVars(n, n, m, vtype=GRB.BINARY, name='w')
    y = model.addVars(n, n, m, vtype=GRB.BINARY, name='y')

    # Objective Function: minimize total EFX-violation count sum y_{i,j,k}.
    model.setObjective(gp.quicksum(y[i, j, g] for i in range(n) for j in range(n) if i != j for g in range(m)), GRB.MINIMIZE)
    # eq:assignment -- each good assigned to exactly one agent.
    model.addConstrs(x.sum("*", g) == 1 for g in range(m))
    model.addConstrs(y[i, j, g] <= x[j, g] for i in range(n) for j in range(n) if i != j for g in range(m))
    model.addConstrs(y[i, j, g] <= w[i, j, g] for i in range(n) for j in range(n) if i != j for g in range(m))
    model.addConstrs(y[i, j, g] >= x[j, g] + w[i, j, g] - 1 for i in range(n) for j in range(n) if i != j for g in range(m))
    # eq:e -- big-M definition forcing w_{i,j,k}=1 whenever agent i strictly
    # prefers X_j \ {k} over their own bundle (see the G_i derivation in the
    # ILP section).
    model.addConstrs(
        gp.quicksum((x[j, k] - x[i, k]) * v[i][k] for k in range(m)) - w[i, j, g] * M[i] <= v[i][g]
        for i in range(n) for j in range(n) if i != j for g in range(m)
    )

    model.optimize()
    status = _solve_status(model)
    if model.SolCount == 0:
        return None, None, status, None, model.Runtime
    allocation = _extract_allocation(x, n, m)
    return allocation, model.ObjVal, status, model.MIPGap, model.Runtime


def solve_ilp_2(n, m, v, M, env, time_limit=None, mip_gap=None):
    """ILP 2: Total Envy (Section 3.2). Minimizes sum_{i,j,k} e_{i,j,k},
    the continuous counterpart of ILP 1's y_{i,j,k} (constraints
    eq:assignment2/eq:e2).
    """
    model = gp.Model("Proximity to EFX", env=env)
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap

    # Decision variables: x_{i,k}, and continuous envy e_{i,j,k} >= 0.
    x = model.addVars(n, m, vtype=GRB.BINARY, name='x')
    e = model.addVars(n, n, m, lb=0, vtype=GRB.CONTINUOUS, name='e')

    # Objective Function: minimize total envy sum e_{i,j,k}.
    model.setObjective(gp.quicksum(e[i, j, g] for i in range(n) for j in range(n) if i != j for g in range(m)), GRB.MINIMIZE)
    # eq:assignment2.
    model.addConstrs(x.sum("*", g) == 1 for g in range(m))
    # eq:e2 -- big-M envy bound; reduces to e_{i,j,k} >= v_i(X_j \ {k}) - v_i(X_i)
    # when k is in X_j.
    model.addConstrs(
        gp.quicksum((x[j, k] - x[i, k]) * v[i][k] for k in range(m)) - v[i][g] - M[i] * (1 - x[j, g]) <= e[i, j, g]
        for i in range(n) for j in range(n) if i != j for g in range(m)
    )

    model.optimize()
    status = _solve_status(model)
    if model.SolCount == 0:
        return None, None, status, None, model.Runtime
    allocation = _extract_allocation(x, n, m)
    return allocation, model.ObjVal, status, model.MIPGap, model.Runtime


def solve_ilp_3(n, m, v, M, env, time_limit=None, mip_gap=None):
    """ILP 3: Egalitarian Envy (Section 3.2) -- the egalitarian
    counterpart of ILP 2, minimizing the worst single agent's total
    envy z instead of the summed total (constraints
    eq:assignment3/eq:e3/eq:z3).
    """
    model = gp.Model("Egalitarian Envy minimization", env=env)
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap

    # Decision variables: x_{i,k}, envy e_{i,j,k} as in ILP 2, and z = worst agent's total envy.
    x = model.addVars(n, m, vtype=GRB.BINARY, name='x')
    e = model.addVars(n, n, m, lb=0, vtype=GRB.CONTINUOUS, name='e')
    z = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='z')

    # Objective Function: minimize z.
    model.setObjective(z, GRB.MINIMIZE)
    # eq:assignment3.
    model.addConstrs(x.sum("*", g) == 1 for g in range(m))
    # eq:e3 (identical to ILP 2's eq:e2).
    model.addConstrs(
        gp.quicksum((x[j, k] - x[i, k]) * v[i][k] for k in range(m)) - v[i][g] - M[i] * (1 - x[j, g]) <= e[i, j, g]
        for i in range(n) for j in range(n) if i != j for g in range(m)
    )
    # eq:z3 -- forces z >= agent i's own total envy, for every i.
    model.addConstrs(z >= gp.quicksum(e[i, j, g] for j in range(n) if j != i for g in range(m)) for i in range(n))

    model.optimize()
    status = _solve_status(model)
    if model.SolCount == 0:
        return None, None, status, None, model.Runtime
    allocation = _extract_allocation(x, n, m)
    return allocation, model.ObjVal, status, model.MIPGap, model.Runtime


def solve_ilp_4(n, m, v, M, env, time_limit=None, mip_gap=None):
    """ILP 4: Egalitarian EFX Violation Count (Section 3.2) -- the
    egalitarian counterpart of ILP 1, minimizing the worst single
    agent's violation count z (constraints
    eq:assignment4/eq:b4/eq:c4/eq:d4/eq:e4/eq:zbound4).
    """
    model = gp.Model("Egalitarian EFX Violations minimization", env=env)
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    if mip_gap is not None:
        model.Params.MIPGap = mip_gap

    # Decision variables: x_{i,k}, w_{i,j,k}, y_{i,j,k} as in ILP 1, and z = worst agent's violation count.
    x = model.addVars(n, m, vtype=GRB.BINARY, name='x')
    w = model.addVars(n, n, m, vtype=GRB.BINARY, name='w')
    y = model.addVars(n, n, m, vtype=GRB.BINARY, name='y')
    z = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name='z')

    # Objective Function: minimize z.
    model.setObjective(z, GRB.MINIMIZE)
    # eq:assignment4.
    model.addConstrs(x.sum("*", g) == 1 for g in range(m))
    # eq:e4 (identical to ILP 1's eq:e).
    model.addConstrs(
        gp.quicksum((x[j, k] - x[i, k]) * v[i][k] for k in range(m)) - w[i, j, g] * M[i] <= v[i][g]
        for i in range(n) for j in range(n) if i != j for g in range(m)
    )
    # eq:b4/eq:c4/eq:d4 (identical to ILP 1) -- force y_{i,j,k} = x_{j,k} AND w_{i,j,k}.
    model.addConstrs(y[i, j, g] <= x[j, g] for i in range(n) for j in range(n) if i != j for g in range(m))
    model.addConstrs(y[i, j, g] <= w[i, j, g] for i in range(n) for j in range(n) if i != j for g in range(m))
    model.addConstrs(y[i, j, g] >= x[j, g] + w[i, j, g] - 1 for i in range(n) for j in range(n) if i != j for g in range(m))
    # eq:zbound4 -- forces z >= agent i's own violation count, for every i.
    model.addConstrs(z >= gp.quicksum(y[i, j, g] for j in range(n) if j != i for g in range(m)) for i in range(n))

    model.optimize()
    status = _solve_status(model)
    if model.SolCount == 0:
        return None, None, status, None, model.Runtime
    allocation = _extract_allocation(x, n, m)
    return allocation, model.ObjVal, status, model.MIPGap, model.Runtime