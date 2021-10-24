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
import popmodel #contains functions for simulation



# allele frequencies import
af = pd.read_excel("./Lynx_AlleleFrequencies_Sep2021.xlsx", engine='openpyxl')
allele_freqs = af.groupby(["Population","Marker"])["Frequency"].agg(lambda x: list(x))


## REPRODUCTION
# Pick a male lynx. A male can mate more then once per generation.
def lynxFather(pop, subPop):
    all_males = [x for x in pop.individuals(subPop) if x.sex() == sim.MALE and x.age >= 2] # all males of reproductive age (older then 2 years)
    while True:
        pick_male = np.random.randint(0, (len(all_males))) # select a random male
        male = all_males[pick_male]
        yield male

#Pick a female lynx. Female can mate only if she has no kittens.
def lynxMother(pop, subPop):
    all_females = [x for x in pop.individuals(subPop) if x.sex() == sim.FEMALE and x.age >= 2] # All females of reproductive age (older then 2 years)

    #print("lynxMother: number of suitable females: {}".format(len(all_females)))
    while True:
        pick_female = np.random.randint(0, (len(all_females)))
        female = all_females[pick_female]
        yield female

def demoModel(gen, pop):
    sim.stat(pop, popSize=True, subPops=[0], suffix="_a")  # Dinaric subpop size
    sim.stat(pop, popSize=True, subPops=[(0, 2)], suffix="_din_old_m")
    sim.stat(pop, popSize=True, subPops=[(0, 5)], suffix="_din_old_f")
    sim.stat(pop, popSize=True, subPops=[1], suffix="_b")  # Slovak subpop size

    fems_din = [x for x in pop.individuals(0) if x.sex() == sim.FEMALE and x.age >= 2]

    if len(fems_din) == 0:
        size_f_din = 0
    else:
        size_f_din = int(litter_size * len(fems_din))# * surivival_rate_kittens  #number of offspring in  Dinaric population
    d_pop_size = pop.dvars().popSize_a - pop.dvars().popSize_din_old_m - pop.dvars().popSize_din_old_f + size_f_din
    if d_pop_size > carrying_capacity:
        d_pop_size = carrying_capacity
    return [d_pop_size, pop.dvars().popSize_b]


def simulation(iterations, years, n_Din, Mortality, TransMales, TransFem, TransFreq=1,TransStart=1,TransEnd=1, allele_freqs = allele_freqs, B=6 ):
    x = []  # empty list to store statistics
    for i in range(iterations):
        pop = sim.Population(size = [n_Din, 5000], loci=[1]*19, #for now number of loci and subpopnames must be the same as in empirical data
                                 infoFields = ["age",'ind_id', 'father_id', 'mother_id', "mating",'migrate_to', "IBD"],
                                 subPopNames = ["Dinaric", "Carpathian"])
        # Set age for Dinaric population
        sim.initInfo(pop = pop, values = list(map(int, np.random.negative_binomial(n = 0.8, p = 0.27, size=n_Din))), # check
                     infoFields="age", subPops = [0])
        # Set age for Slovak population randomly in interval from 1 to 5 years
        sim.initInfo(pop = pop, values = list(map(int, np.random.randint(1, 5 +1, 1500))),
                     infoFields="age", subPops = [1])
        # Initialize genotypes
        for i in range(len(pop.numLoci())):
            for name in pop.subPopNames():
                sim.initGenotype(pop, prop=allele_freqs[name][i], loci=i, subPops=[name]) # freq for allele frequencies

        # We need VirtualSplitter to transfer non-reproductive animals to the new generation
        pop.setVirtualSplitter(sim.CombinedSplitter([
            sim.ProductSplitter([
                sim.SexSplitter(),
                sim.InfoSplitter(field = "age", cutoff = [2,13])])])) # add vspMap, if needed

        pop.evolve(
        initOps=[
            sim.InitSex(),
            sim.InitLineage(),
            sim.IdTagger(), # needed for IBD
        ],
        # increase the age of everyone by 1 before mating only in Dinaric population.
        preOps=[sim.InfoExec('age += 1', subPops=[0]),
                sim.PyOperator(func=popmodel.NaturalMortality, subPops=[0], param={"Mortality": Mortality}), # apply only to Dinaric
                # Different translocation scenarios
                sim.Migrator(rate=[
                        [TransMales,0], # in the migration matrix columns = populations, rows = subpopulations, position - where to migrate
                        [TransFem,0]],
                         mode=sim.BY_COUNTS,
                         subPops=[(1,1),(1,4)], #from Carpathian population to Dinaric.
                         step = TransFreq, begin=TransStart, end=TransEnd),
                 #sim.Stat(effectiveSize=sim.ALL_AVAIL, subPops=[(0,1),(0,2),(0,4), (1,1), (1,2), (1,4)], vars='Ne_demo_base'),
                 #sim.Stat(effectiveSize=sim.ALL_AVAIL,subPops=[(0,1),(0,2),(0,4), (1,1), (1,2), (1,4)], vars='Ne_demo_base_sp')
                ],
        matingScheme=sim.HeteroMating([
            # CloneMating will keep individual sex and all
            # information fields (by default). Used to keep an age structure.
            # The age of offspring will be zero.

            sim.HomoMating(subPops=sim.ALL_AVAIL,
                chooser=sim.CombinedParentsChooser(
                    fatherChooser=sim.PyParentsChooser(generator=lynxFather),
                    motherChooser=sim.PyParentsChooser(generator=lynxMother)
                ),
                generator=sim.OffspringGenerator(ops=[
                    sim.InfoExec("age = 0"),
                    sim.IdTagger(),
                    sim.PedigreeTagger(),
                    #sim.ParentsTagger(),
                    sim.MendelianGenoTransmitter()
                ], numOffspring=(sim.UNIFORM_DISTRIBUTION, 1, 4))),
            sim.CloneMating(subPops=[(0,0), (0,1), (0,3), (0,4), 1], weight=-1),

        ], subPopSize=demoModel), #Demomodel can be used for abundancy estimates

        postOps = [
            sim.PyOperator(func=popmodel.IBD, subPops=[0],begin=10),
            sim.PyOperator(func=popmodel.InbreedingLoad, subPops=[0],begin=10, param={"B": B}), # apply only to Dinaric
            sim.PyOperator(func=popmodel.CalcStats, param={"x": x}, begin=10),  # int(0.2*generations)
                   ],
        gen = years
    )

    x = pd.concat(x)
    return x


#Baseline values

iterations = 10 #number of iterations
years = 20 # number of years in each iteration
n_Din = 100 # number of animals in Dinaric populaiton
## Migration proportions
TransFreq = 1 # per one year, minimum value is 1
TransMales = 0 # number of translocated males per year
TransFem = 0 #number of translocated females per year
TransStart = 1
TransEnd = 1
Mortality = 0.19

# these all global variables
litter_size = 1.9
surivival_rate_kittens = 0.6 #calculated for Canadian lynx
carrying_capacity = 150
Stats = simulation(iterations, years, n_Din, Mortality, TransMales, TransFem, TransFreq,TransStart,TransEnd, allele_freqs)


