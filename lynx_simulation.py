#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 09:46:09 2020

@author: elena
"""

import simuOpt
simuOpt.setOptions(alleleType='lineage', quiet=True)
import simuPOP as sim
import pandas as pd
import numpy as np
import seaborn as sns
import popmodel #contains all functions for simulation
import random as random

### FUNCTIONS ###

## REPRODUCTION

# Pick a male lynx
def lynxFather(pop, subPop):
    all_males = [x for x in pop.individuals(subPop) if x.sex() == sim.MALE and x.age >= 2]
    iterator = 0
    while True:
        pick_male = np.random.randint(0, (len(all_males)))
        male = all_males[pick_male]
        yield male

def lynxMother(pop, subPop):
    all_females = [x for x in pop.individuals(subPop) if x.sex() == sim.FEMALE and x.hc == 0 and x.age >= 2]

    #print("lynxMother: number of suitable females: {}".format(len(all_females)))
    while True:
        pick_female = np.random.randint(0, (len(all_females)))
        female = all_females[pick_female]
        female.hc = 1
        yield female

# Number of kittens
def demoModel(gen, pop):
    sim.stat(pop, popSize=True, subPops=[0], suffix="_a")  # Dinaric subpop size
    sim.stat(pop, popSize=True, subPops=[(0, 2)], suffix="_din_old_m")
    sim.stat(pop, popSize=True, subPops=[(0, 5)], suffix="_din_old_f")
    sim.stat(pop, popSize=True, subPops=[1], suffix="_b")  # Slovak subpop size

    fems_din = [x for x in pop.individuals(0) if x.sex() == sim.FEMALE and x.hc == 0 and x.age >= 2]

    if len(fems_din) == 0:
        size_f_din = 0
    else:
        size_f_din = int(1.9 * len(fems_din))  # * 0.8  #CHECK COEFF  0.8 = survival rate of cubs, 1.9 - litter size

    return [pop.dvars().popSize_a - pop.dvars().popSize_din_old_m - pop.dvars().popSize_din_old_f + size_f_din,
            pop.dvars().popSize_b]

## Mortality

def NaturalMortality(pop):
    all_inds = [x for x in pop.individuals(0)]
    sampling_m = random.choices(all_inds, k=int(0.02 * len(all_inds))) #Adult mortality 0.19-0.29
    out_ids = [x.ind_id for x in sampling_m]
    pop.removeIndividuals(IDs=out_ids, idField="ind_id")
    return True


def simulation(iterations, generations, TransFreq, TransMales, TransFem):
    x = []  # empty list to store LD Ne calculations
    Ne = [] # empty list to store demographic Ne calculations
    
    for i in range(iterations):
        pop = sim.Population(size = [71, 1500], loci=[1]*20,
                                 infoFields = ["age",'ind_id', 'father_idx', 'mother_idx', "mating", "hc",'migrate_to'],
                                 subPopNames = ["Dinaric", "Slovak"])
        # Set age for Dinaric population
        sim.initInfo(pop = pop, values = list(map(int, np.random.negative_binomial(n = 1, p = 0.25, size=71))),
                     infoFields="age", subPops = [0])
        # Set age for Slovak population randomly in interval from 1 to 5 years
        sim.initInfo(pop = pop, values = list(map(int, np.random.randint(1, 5 +1, 1500))),
                     infoFields="age", subPops = [1])

        # We need VirtualSplitter to transfer non-reproductive animals to the new generation

        pop.setVirtualSplitter(sim.CombinedSplitter([
            sim.ProductSplitter([
                sim.SexSplitter(),
                sim.InfoSplitter(field = "age", cutoff = [2,13])])])) # add vspMap, if needed
        pop.evolve(
            initOps=[
                sim.InitSex(),
                # TO DO: separately for Din and Slovak
                # genotype from empirical allele frequencies and number of alleles
                sim.InitGenotype(freq=[0.2, 0.7, 0.1]),
                # assign an unique ID to everyone.
                sim.IdTagger(),
            ],
            # increase the age of everyone by 1 before mating only in Dinaric population.
            preOps=[sim.InfoExec('age += 1', subPops=[0]),
                    sim.PyOperator(func=NaturalMortality, subPops=[0]), # apply only to Dinaric
                    sim.InfoExec("hc +=1 if 0 < hc < 2  else 0"), # Females can have cubs ones per 2 years
                    # Different translocation scenarios
                    sim.Migrator(rate=[
                            [TransMales,0], # Columns = populations, rows = subpopulations
                            [TransFem,0]],
                             mode=sim.BY_COUNTS,
                             subPops=[(1,1),(1,4)], #from Slovak population to Dinaric.
                             step = TransFreq),
                     sim.Stat(effectiveSize=sim.ALL_AVAIL, subPops=[(0,1),(0,2),(0,4), (1,1), (1,2), (1,4)], vars='Ne_demo_base'),
                     sim.Stat(effectiveSize=sim.ALL_AVAIL,subPops=[(0,1),(0,2),(0,4), (1,1), (1,2), (1,4)], vars='Ne_demo_base_sp')
                    #sim.PyEval(r'"Cro %d, Slo %d' ' % (Cro, Slo)', "Cro = pop.subPopSize(0)" "Slo = pop.subPopSize(1)",exposePop='pop'),
                    ],
            matingScheme=sim.HeteroMating([
                # CloneMating will keep individual sex and all
                # information fields (by default).
                # The age of offspring will be zero.

                sim.HomoMating(subPops=sim.ALL_AVAIL,
                    chooser=sim.CombinedParentsChooser(
                        fatherChooser=sim.PyParentsChooser(generator=lynxFather),
                        motherChooser=sim.PyParentsChooser(generator=lynxMother)
                    ),
                    generator=sim.OffspringGenerator(ops=[
                        sim.InfoExec("age = 0"),
                        sim.IdTagger(),
                        #sim.PedigreeTagger(),
                        sim.ParentsTagger(),
                        sim.MendelianGenoTransmitter()
                    ], numOffspring=(sim.UNIFORM_DISTRIBUTION, 1, 3))),
                sim.CloneMating(subPops=[(0,0), (0,1), (0,3), (0,4), 1], weight=-1),

            ], subPopSize=demoModel), #Demomodel can be used for abundancy estimates

            postOps = [
                sim.PyOperator(func = popmodel.CalcNe, param={"me":me, "Ne":Ne}, begin=int(0.2*generations)),
                sim.PyOperator(func = popmodel.CalcLDNe, param={"me":me, "x":x}, begin=int(0.2*generations))
                       ],

            gen = generations
        )
    x = pd.concat(x)
    Ne = pd.concat(Ne)
    x.loc[x["population"] ==0,"population"] = "cro"
    x.loc[x["population"] ==1,"population"] = "slo"
    x = x[x['cutoff'] == 0]
    x = x.rename(columns={0: "Ne"})
    return x, Ne


#Baseline values

#success_repr_males and success_dominant_males must be global variables, so it need to be set by the same name as here
success_repr_males = 1 # Chance of reproductive males to have an offspring (p in binomial distribution)
success_dominant_males = 1 # Chance of dominant males to have an offspring (p in binomial distribution)


iterations = 50 #number of iterations
generations = 50 # number of generations in each iteration

## Migration proportions
cro_to_slo = 0 #proportion of reproductive males, migrating from Croatia to Slovenia
slo_to_cro = 0 # and from Slovenia to Croatia.
TransFreq = 1 # one time per year
TransMales = 10 #one animal per year
TransFem = 10






#The function returns two dataframes: the first is Ne LD and the second is Ne demographic (EXACTLY THIS ORDER).
Ne_LD_baseline, Ne_demo_baseline = simulation(me, iterations, generations, cro_to_slo, slo_to_cro, slo_cull, cro_cull)

#Plot demographic (direct) effecitve population size dynamics
sns.relplot(x="gen", y="Ne", col = "me", hue="population", kind="line", data=Ne_demo_baseline)

#Plot linkage disequilibrium effecitve population size dynamics
sns.relplot(x="gen", y="Ne", col = "me", hue="population", kind="line", data=Ne_LD_baseline)





