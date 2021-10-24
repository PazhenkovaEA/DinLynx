# Estimate population size after mating (All individuals - N of 20 y.o. males - N. of 20 y.o. females + number of cubs)
# Number of cubs is calculated as number of females older than 2 y.o. and without cubs * 1.9 - mean litter size from Bischof et al (2009)
# !!! implement density-dependant population growth model
import simuPOP as sim
import numpy as np
#import random as random

def demoModel(gen, pop):
    sim.stat(pop, popSize=True, subPops=[0], suffix = "_a") # Croatian subpop size
    sim.stat(pop, popSize=True, subPops=[(0, 3)], suffix="_Cro_old_m")
    sim.stat(pop, popSize=True, subPops=[(0, 5)], suffix="_Cro_old_f")
    sim.stat(pop, popSize=True, subPops=[1], suffix="_b") # Slovenian subpop size
    sim.stat(pop, popSize=True, subPops=[(1, 3)], suffix="_Slo_old_m")
    sim.stat(pop, popSize=True, subPops=[(1, 5)], suffix="_Slo_old_f")
    fems_slo = [x for x in pop.individuals(1) if x.sex() == sim.FEMALE and x.hc == 0 and x.age >= 3] #hc = has cubs
    if len(fems_slo) == 0:
        size_f_slo =0
    else:
        size_f_slo = int(1.9 * len(fems_slo)) #* 0.8 # 0.8 = survival rate of cubs, 1.9 - litter size from Bischof et al (2009)


    fems_cro = [x for x in pop.individuals(0) if x.sex() == sim.FEMALE and x.hc == 0 and x.age >= 3]

    if len(fems_cro) == 0:
        size_f_cro = 0
    else:
        size_f_cro = int(1.9 * len(fems_cro))# * 0.8
        
        
    return [pop.dvars().popSize_a - pop.dvars().popSize_Cro_old_m - pop.dvars().popSize_Cro_old_f + size_f_cro, pop.dvars().popSize_b - pop.dvars().popSize_Slo_old_m - pop.dvars().popSize_Slo_old_f + size_f_slo]



'''
Function will pick a father bear. Bears are sorted according to age (size) and
oldest males get to mate "me" (mating events) times.

@author Roman Luštrik
'''
def bearFather(pop, subPop, param):

    all_males = [x for x in pop.individuals(subPop) if x.sex() == sim.MALE and x.age >= 3]

    # sort them by age
    #print("bearFather: number of suitable males: {}".format(len(all_males)))
    # print([x.age for x in all_males])
    all_males = sorted(all_males, key = lambda x: x.age, reverse = True)

   # Mate each male `me` times.
    while True:
        for i in range(param["me"]):
            yield all_males[0]
            # print("male of age: {}".format(all_males[0].age))
        all_males = all_males[1:] # after mating "me" times, remove animal from the pool

'''
Function picks mothers if bear is female, has no cubs and is older or equal to 3 years.

@author Roman Luštrik
'''

def bearMother(pop, subPop):
    # prepare females in estrus
    all_females = [x for x in pop.individuals(subPop) if x.sex() == sim.FEMALE and x.hc == 0 and x.age >= 3]

    #print("bearMother: number of suitable females: {}".format(len(all_females)))

    while True:
        pick_female = np.random.randint(0, (len(all_females)))
        female = all_females[pick_female]
        female.hc = 1
        yield female
