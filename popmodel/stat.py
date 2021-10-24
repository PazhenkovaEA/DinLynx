# -*- coding: utf-8 -*-
# Function to calculate Ne as described in (Waples, 2006) (the same as in Ne estimator v. 2 software)
import simuPOP as sim
import pandas as pd
import numpy as np
from statistics import mean 

def CalcStats(pop, param):
    sim.stat(pop = pop, popSize = True,  heteroFreq=sim.ALL_AVAIL,
             subPops=[0]) # calculate number of animals, IBS, IBD and observed heterozygosity for Din pop
    H_obs_din = np.mean(list(pop.vars()["heteroFreq"].values()))  # observed heterozygosity, multiple loci
    sim.stat(pop, alleleFreq=sim.ALL_AVAIL, subPops=[0])
    H_exp_din = 1- np.mean([sum([x*x for x in pop.vars()["alleleFreq"][i].values()])
                         for i in range(len(pop.vars()["alleleFreq"]))]) #expected heterozygosity for multiple loci
    sim.stat(pop, alleleFreq=sim.ALL_AVAIL, subPops=[1])
    H_exp_slovak = 1 - np.mean([sum([x * x for x in pop.vars()["alleleFreq"][i].values()])
                         for i in range(len(pop.vars()["alleleFreq"]))])  # expected heterozygosity for multiple loci
    F_eff = 1 - H_exp_din/H_exp_slovak
    relative_fitness = 1 -np.e**(-6*F_eff)
    a = pd.DataFrame({'Generation': [pop.dvars().gen - 5], 'N_din_a': [pop.dvars().subPopSize[0]],
                  "Het_exp_din": [H_exp_din], "H_exp_slovak": [H_exp_slovak],"Het_obs_din": [H_obs_din], "F_eff": [F_eff],
                   "relative_fitness" : [relative_fitness]})
    param["x"].append(a)
    return True

def CalcStats1(pop, param):
    sim.stat(pop = pop, popSize = True,
             inbreeding=sim.ALL_AVAIL, subPops=[0]) # calculate number of animals, IBS, IBD and observed heterozygosity for Din pop
    sim.stat(pop=pop, effectiveSize=sim.ALL_AVAIL, vars="Ne_LD_sp")  # calculate Ne for both populations separately
    IBD_din = np.mean(list(pop.vars()["IBD_freq"].values()))
    #ibd = [ind.IBD for ind in pop.individuals(subPop=[0])]

    a = pd.DataFrame({'N_din_b': [pop.dvars().subPopSize[0]],
                  "Ne_LD": [pop.dvars(0).Ne_LD[0.01]],  # 0.01 threshold
                  "IBD_din": [IBD_din]})
    param["y"].append(a)
    return True