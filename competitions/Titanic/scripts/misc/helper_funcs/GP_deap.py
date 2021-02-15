# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 09:23:17 2021

@author: oislen
"""

import string
import numpy as np

# now this number indicates the number of generations, which can quite long.
# I recommend something around the number of 1000, but 100 is timewise okay (~5 min)
# Currently it is at 10 to reduce runtime. Of course the more iterations the better the algorithm
# Some code taken from: https://github.com/innjoshka/Genetic-Programming-Titanic-Kaggle
HOWMANYITERS = 10

def GP_deap(evolved_train):
    global HOWMANYITERS
    import operator
    import math
    import random


    from deap import algorithms
    from deap import base, creator
    from deap import tools
    from deap import gp

    # dropping Survived and Passenger ID because we can not use them for training
    outputs = evolved_train['Survived'].values.tolist()
    evolved_train = evolved_train.drop(["Survived","PassengerId"],axis=1)
    inputs = evolved_train.values.tolist() # to np array
    


    def protectedDiv(left, right):
        try:
            return left / right
        except ZeroDivisionError:
            return 1

    def randomString(stringLength=10):
        """Generate a random string of fixed length """
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(stringLength))
    #choosing Primitives
    pset = gp.PrimitiveSet("MAIN", len(evolved_train.columns))  # add here
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(math.cos, 1)
    pset.addPrimitive(math.sin, 1)
    pset.addPrimitive(math.tanh,1)
    pset.addPrimitive(max, 2)
    pset.addPrimitive(min, 2)
    pset.addEphemeralConstant(randomString(), lambda: random.uniform(-10,10))
    # 50 as a precaution. 34 would be enough
    pset.renameArguments(ARG0='x1')
    pset.renameArguments(ARG1='x2')
    pset.renameArguments(ARG2='x3')
    pset.renameArguments(ARG3='x4')
    pset.renameArguments(ARG4='x5')
    pset.renameArguments(ARG5='x6')
    pset.renameArguments(ARG6='x7')
    pset.renameArguments(ARG7='x8')
    pset.renameArguments(ARG8='x9')
    pset.renameArguments(ARG9='x10')
    pset.renameArguments(ARG10='x11')
    pset.renameArguments(ARG11='x12')
    pset.renameArguments(ARG12='x13')
    pset.renameArguments(ARG13='x14')
    pset.renameArguments(ARG14='x15')
    pset.renameArguments(ARG15='x16')
    pset.renameArguments(ARG16='x17')
    pset.renameArguments(ARG17='x18')
    pset.renameArguments(ARG18='x19')
    pset.renameArguments(ARG19='x20')
    pset.renameArguments(ARG20='x21')
    pset.renameArguments(ARG21='x22')
    pset.renameArguments(ARG22='x23')
    pset.renameArguments(ARG23='x24')
    pset.renameArguments(ARG24='x25')
    pset.renameArguments(ARG25='x26')
    pset.renameArguments(ARG26='x27')
    pset.renameArguments(ARG27='x28')
    pset.renameArguments(ARG28='x29')
    pset.renameArguments(ARG29='x30')
    pset.renameArguments(ARG30='x31')
    pset.renameArguments(ARG31='x32')
    pset.renameArguments(ARG32='x33')
    pset.renameArguments(ARG33='x34')
    pset.renameArguments(ARG34='x35')
    pset.renameArguments(ARG35='x36')
    pset.renameArguments(ARG36='x37')
    pset.renameArguments(ARG37='x38')
    pset.renameArguments(ARG38='x39')
    pset.renameArguments(ARG39='x40')
    pset.renameArguments(ARG40='x41')
    pset.renameArguments(ARG41='x42')
    pset.renameArguments(ARG42='x43')
    pset.renameArguments(ARG43='x44')
    pset.renameArguments(ARG44='x45')
    pset.renameArguments(ARG45='x46')
    pset.renameArguments(ARG46='x47')
    pset.renameArguments(ARG47='x48')
    pset.renameArguments(ARG48='x49')
    pset.renameArguments(ARG49='x50')

    # two object types is needed: an individual containing the genotype
    # and a fitness -  The reproductive success of a genotype (a measure of quality of a solution)
    creator.create("FitnessMin", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)


    #register some parameters specific to the evolution process.
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3) #
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)


    #evaluation function, which will receive an individual as input, and return the corresponding fitness.
    def evalSymbReg(individual):
        # Transform the tree expression in a callable function
        func = toolbox.compile(expr=individual)
        # Evaluate the accuracy of individuals // 1|0 == survived
        return math.fsum(np.round(1.-(1./(1.+np.exp(-func(*in_))))) == out for in_, out in zip(inputs, outputs)) / len(evolved_train),


    toolbox.register("evaluate", evalSymbReg)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=3)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)

    #Statistics over the individuals fitness and size
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    stats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)


    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.3, ngen=HOWMANYITERS, stats=stats,
                                   halloffame=hof, verbose=True)

    #Parameters:
    #population – A list of individuals.
    #toolbox – A Toolbox that contains the evolution operators.
    #cxpb – The probability of mating two individuals.
    #mutpb – The probability of mutating an individual.
    #ngen – The number of generation.
    #stats – A Statistics object that is updated inplace, optional.
    #halloffame – A HallOfFame object that will contain the best individuals, optional.
    #verbose – Whether or not to log the statistics.

    # Transform the tree expression of hof[0] in a callable function and return it
    func2 = toolbox.compile(expr=hof[0]) 

    return func2