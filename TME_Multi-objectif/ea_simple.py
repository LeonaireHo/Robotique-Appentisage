import numpy as np
from deap import base, creator, benchmarks

import random
from deap import tools

# ne pas oublier d'initialiser la grane aléatoire (le mieux étant de le faire dans le main))
random.seed()


def fit_25(array):
    return np.percentile(array,25)
    
def fit_75(array):
    return np.percentile(array,75)

def ea_simple(n, nbgen, evaluate, IND_SIZE, weights=(-1.0,)):
    """Algorithme evolutionniste elitiste

    Algorithme evolutionniste elitiste. 
    :param n: taille de la population
    :param nbgen: nombre de generation 
    :param evaluate: la fonction d'évaluation
    :param IND_SIZE: la taille d'un individu
    :param weights: les poids à utiliser pour la fitness (ici ce sera (-1.0,) pour une fonction à minimiser et (1.0,) pour une fonction à maximiser)
    """
    #code
    mini = -5
    maxi = 5
    mute_type = 15.0
    CXPB, MUTPB, NGEN = 0.3, 0.2, nbgen
    #code2

    creator.create("FitnessMin", base.Fitness, weights=weights)
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # à compléter pour sélectionner les opérateurs de mutation, croisement, sélection avec des toolbox.register(...)

    #code 
    
    toolbox.register("mutate", tools.mutPolynomialBounded, eta = mute_type, up = maxi, low = mini, indpb = MUTPB)
    
    toolbox.register("mate", tools.cxSimulatedBinary, eta = 0.5)

    toolbox.register("select", tools.selTournament, tournsize=3)


    #population et initialisation
    toolbox.register("attribute", random.uniform,mini, maxi)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=IND_SIZE)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)



    #code2

    # Les statistiques permettant de récupérer les résultats
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("fit_25", fit_25)
    stats.register("fit_75", fit_75)

    # La structure qui permet de stocker les statistiques
    logbook = tools.Logbook()


    # La structure permettant de récupérer le meilleur individu
    hof = tools.HallOfFame(1)


    ## à compléter pour initialiser l'algorithme, n'oubliez pas de mettre à jour les statistiques, le logbook et le hall-of-fame.
    #code
    toolbox.register("evaluate", evaluate)
    
    #initialisation de la population
    pop = toolbox.population(n)
    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    #code2

    for gen in range(nbgen):
        # if (gen%10==0):
        #     print("+",end="", flush=True)
        # else:
        #     print(".",end="", flush=True)

        ## à compléter en n'oubliant pas de mettre à jour les statistiques, le logbook et le hall-of-fame
        #code 
        stat = stats.compile(pop)
        logbook.record(gen=gen, **stat)

        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = [toolbox.clone(offspring[i]) for i in range(len(offspring))]


        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring
        #update the hall of fame
        hof.update(pop)

    #statistiques sur la population
    stat = stats.compile(pop)
    #logbook
    logbook.record(gen=nbgen, **stat)
        #code2
        
    return pop, hof, logbook
