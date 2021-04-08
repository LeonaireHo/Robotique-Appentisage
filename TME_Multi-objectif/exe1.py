
# Note: l'import d'un fichier ne se fait qu'une seule fois. Si vous modifiez ce fichier,
# il vous faut redémarrer votre kernel si vous voulez prendre en compte les modifications.
# vous pouvez éviter cela de la façon suivante:
import importlib # une seule fois

#import mon_module_python # le module doit avoir été importé une première fois
#importlib.reload(mon_module_python) # cette ligne permet de charger la dernière version

import matplotlib.pyplot as plt

from ea_simple import ea_simple
import deap
import numpy as np


for i in [5, 10, 100, 200]:
    nums_moy = []
    nums_fit25 = []
    nums_fit75 = []
    for j in range(10):
        pop, hof, logbook = ea_simple(i, 10, deap.benchmarks.ackley, 10)
        gen, moyenne, fit_25, fit_75 = logbook.select("gen", "avg", "fit_25", "fit_75")
        nums_moy.append(moyenne)
        nums_fit25.append(fit_25)
        nums_fit75.append(fit_75)

    moyenne = []
    for k in range(len(nums_moy[0])):
        temp = []
        for l in nums_moy:
            temp.append(l[k])
        moyenne.append(np.median(temp))

    fit_25 = []
    for k in range(len(nums_fit25[0])):
        temp = []
        for l in nums_fit25:
            temp.append(l[k])
        fit_25.append(np.median(temp))

    fit_75 = []
    for k in range(len(nums_fit75[0])):
        temp = []
        for l in nums_fit75:
            temp.append(l[k])
        fit_75.append(np.median(temp))

    plt.plot(gen, moyenne,label = "Fit_"+i.__str__())
    plt.fill_between(gen, fit_25, fit_75, alpha=0.25, linewidth=0)
    plt.legend(loc='upper left')
    plt.savefig("exe1")