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


# allele frequencies import
af = pd.read_excel("/Users/elena/Desktop/Lynx_AlleleFrequencies_Sep2021.xlsx", engine='openpyxl')
allele_freqs = af.groupby(["Population","Marker"])["Frequency"].agg(lambda x: list(x))

### FUNCTIONS ###

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
    all_females = [x for x in pop.individuals(subPop) if x.sex() == sim.FEMALE and x.hk == 0 and x.age >= 2] # hk = have kittens (0 - no . All females of reproductive age (older then 2 years)

    #print("lynxMother: number of suitable females: {}".format(len(all_females)))
    while True:
        pick_female = np.random.randint(0, (len(all_females)))
        female = all_females[pick_female]
        female.hk = 1
        yield female

#To Do simulate lethal equivalents
def demoModel(gen, pop):
    sim.stat(pop, popSize=True, subPops=[0], suffix="_a")  # Dinaric subpop size
    sim.stat(pop, popSize=True, subPops=[(0, 2)], suffix="_din_old_m")
    sim.stat(pop, popSize=True, subPops=[(0, 5)], suffix="_din_old_f")
    sim.stat(pop, popSize=True, subPops=[1], suffix="_b")  # Slovak subpop size

    fems_din = [x for x in pop.individuals(0) if x.sex() == sim.FEMALE and x.hk == 0 and x.age >= 2]

    if len(fems_din) == 0:
        size_f_din = 0
    else:
        size_f_din = int(litter_size * len(fems_din)) * surivival_rate_kittens  #number of offspring in  Dinaric population
    d_pop_size = pop.dvars().popSize_a - pop.dvars().popSize_din_old_m - pop.dvars().popSize_din_old_f + size_f_din
    if d_pop_size > carrying_capacity:
        d_pop_size = carrying_capacity
    return [d_pop_size, pop.dvars().popSize_b]

## Mortality

def NaturalMortality(pop):
    all_inds = [x for x in pop.individuals(0)]
    sampling_m = random.choices(all_inds, k=int(Mortality * len(all_inds))) #Adult mortality 0.19-0.29
    out_ids = [x.ind_id for x in sampling_m]
    pop.removeIndividuals(IDs=out_ids, idField="ind_id")
    return True



## Statistics
def CalcStats(pop, param):
    sim.stat(pop = pop, popSize = True,  heteroFreq=sim.ALL_AVAIL,
             inbreeding=sim.ALL_AVAIL, subPops=[0]) # calculate number of animals, IBS, IBD and observed heterozygosity for Din pop
    H_obs_din = np.mean(list(pop.vars()["heteroFreq"].values()))  # observed heterozygosity, multiple loci
    sim.stat(pop=pop, effectiveSize=sim.ALL_AVAIL, vars="Ne_LD_sp")  # calculate Ne for both populations separately
    sim.stat(pop, alleleFreq=sim.ALL_AVAIL, subPops=[0])
    H_exp_din = 1- np.mean([sum([x*x for x in pop.vars()["alleleFreq"][i].values()])
                         for i in range(len(pop.vars()["alleleFreq"]))]) #expected heterozygosity for multiple loci
    sim.stat(pop, alleleFreq=sim.ALL_AVAIL, subPops=[1])
    H_exp_slovak = 1 - np.mean([sum([x * x for x in pop.vars()["alleleFreq"][i].values()])
                         for i in range(len(pop.vars()["alleleFreq"]))])  # expected heterozygosity for multiple loci
    F_eff = 1 - H_exp_din/H_exp_slovak
    IBD_din = np.mean(list(pop.vars()["IBD_freq"].values()))
    relative_fitness = 1 -np.e**(-6*F_eff)
    a = pd.DataFrame({'Generation': [pop.dvars().gen], 'N_din': [pop.dvars().subPopSize[0]],
                  "Ne_LD": [pop.dvars(0).Ne_LD[0.01]],  # 0.01 threshold
                  "Het_exp_din": [H_exp_din], "H_exp_slovak": [H_exp_slovak],"Het_obs_din": [H_obs_din], "F_eff": [F_eff],
                  "IBD_din": [IBD_din], "relative_fitness" : [relative_fitness]})
    param["x"].append(a)
    return True




def simulation(iterations, generations, n_Din, TransMales, TransFem, TransFreq=1,TransStart=1, allele_freqs = allele_freqs):
    x = []  # empty list to store statistics
    for i in range(iterations):
        pop = sim.Population(size = [n_Din, 5000], loci=[1]*19, #for now number of loci and subpopnames must be the same as in empirical data
                                 infoFields = ["age",'ind_id', 'father_idx', 'mother_idx', "mating", "hk",'migrate_to'],
                                 subPopNames = ["Dinaric", "Carpathian"])
        # Set age for Dinaric population
        sim.initInfo(pop = pop, values = list(map(int, np.random.negative_binomial(n = 0.8, p = 0.27, size=71))), # check
                     infoFields="age", subPops = [0])
        # Set age for Slovak population randomly in interval from 1 to 5 years
        sim.initInfo(pop = pop, values = list(map(int, np.random.randint(1, 5 +1, 1500))),
                     infoFields="age", subPops = [1])
        # Initialize genotypes - check, if you need to move this chunk
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

                # TO DO: separately for Din and Slovak

                # genotype from empirical allele frequencies and number of alleles
                #sim.InitGenotype(freq=[0.2, 0.7, 0.1]),
                # assign an unique ID to everyone.
                sim.IdTagger(),
            ],
            # increase the age of everyone by 1 before mating only in Dinaric population.
            preOps=[sim.InfoExec('age += 1', subPops=[0]),
                    sim.PyOperator(func=NaturalMortality, subPops=[0]), # apply only to Dinaric
                    sim.InfoExec("hk +=1 if 0 < hk < 2  else 0"), # Females can have kittens ones per 2 years
                    # Different translocation scenarios
                    sim.Migrator(rate=[
                            [TransMales,0], # in the migration matrix columns = populations, rows = subpopulations, position - where to migrate
                            [TransFem,0]],
                             mode=sim.BY_COUNTS,
                             subPops=[(1,1),(1,4)], #from Carpathian population to Dinaric.
                             step = TransFreq, begin=TransStart),
                     sim.Stat(effectiveSize=sim.ALL_AVAIL, subPops=[(0,1),(0,2),(0,4), (1,1), (1,2), (1,4)], vars='Ne_demo_base'),
                     sim.Stat(effectiveSize=sim.ALL_AVAIL,subPops=[(0,1),(0,2),(0,4), (1,1), (1,2), (1,4)], vars='Ne_demo_base_sp')
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
                        #sim.PedigreeTagger(),
                        sim.ParentsTagger(),
                        sim.MendelianGenoTransmitter()
                    ], numOffspring=(sim.UNIFORM_DISTRIBUTION, 1, 4))),
                sim.CloneMating(subPops=[(0,0), (0,1), (0,3), (0,4), 1], weight=-1),

            ], subPopSize=demoModel), #Demomodel can be used for abundancy estimates

            postOps = [
                sim.PyOperator(func = CalcStats, param={"x":x}, begin=int(0.2*generations))
                       ],

            gen = generations
        )
    x = pd.concat(x)
    return x


#Baseline values

iterations = 15 #number of iterations
generations = 20 # number of generations in each iteration
n_Din = 71 # number of animals in Dinaric populaiton
## Migration proportions
TransFreq = 1 # per one year, minimum value is 1
TransMales = 0 # number of translocated males per year
TransFem = 0 #number of translocated females per year
TransStart = 1

# these all global variables
litter_size = 1.9
Mortality = 0.25
surivival_rate_kittens = 0.6 #calculated for Canadian lynx
carrying_capacity = 150
Stats = simulation(iterations, generations, n_Din, TransMales, TransFem, TransFreq,TransStart, allele_freqs)




#Plot
sns.relplot(x="Generation", y="F_eff", kind="line", data=Stats)





#Plot linkage disequilibrium effecitve population size dynamics
sns.relplot(x="gen", y="Ne", col = "me", hue="population", kind="line", data=Ne_LD_baseline)




x = np.random.negative_binomial(n = 0.8, p = 0.27, size=100)
sns.distplot(x)
max(x)