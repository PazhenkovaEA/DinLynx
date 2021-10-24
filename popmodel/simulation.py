#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 09:54:52 2020
]
NOT WORKING CLASS!

@author: elena
"""
import simuPOP as sim
import numpy as np

class Simulation:
    def __init__(self, me, generations, iterations, cro_to_slo, slo_to_cro, slo_cull, cro_cull, success_repr_males, success_dominant_males):
        self.me = me
        self.generations = generations
        self.iterations = iterations
        self.cro_to_slo= cro_to_slo
        self.slo_to_cro = slo_to_cro
        self.slo_cull = slo_cull 
        self.cro_cull = cro_cull
        self.success_repr_males=success_repr_males
        self.success_dominant_males=success_dominant_males
        self.x = []
        self.Ne = []
    def bearFather(self, pop, subPop):

        all_males = [x for x in pop.individuals(subPop) if x.sex() == sim.MALE and x.age >= 3]
    
        # sort them by age
        #print("bearFather: number of suitable males: {}".format(len(all_males)))
        # print([x.age for x in all_males])
        #all_males = sorted(all_males, key = lambda x: x.age, reverse = True)
        while len(all_males) > 0:
            if all_males[0].age < 6:
                mating = np.random.binomial(self.me, self.success_repr_males)
            else:
                mating = np.random.binomial(self.me, self.success_dominant_males)
            if mating !=0:
                for i in range(mating):
                    yield all_males[0]
                    # print("male of age: {}".format(all_males[0].age))
            all_males = all_males[1:] # after mating "me" times, remove animal from the pool

    def bearMother(self, pop, subPop):
        all_females = [x for x in pop.individuals(subPop) if x.sex() == sim.FEMALE and x.hc == 0 and x.age >= 3]
    
        #print("bearMother: number of suitable females: {}".format(len(all_females)))
    
        while True:
            pick_female = np.random.randint(0, (len(all_females)))
            female = all_females[pick_female]
            female.hc = 1
            yield female
    def simulation(self):
        self.pop = sim.Population(size = [500, 500], loci=[1]*20,
                                 infoFields = ["age",'ind_id', 'father_idx', 'mother_idx', "hc", "ywc",'migrate_to'],
                                 subPopNames = ["croatia", "slovenia"])
        sim.initInfo(pop = self.pop, values = list(map(int, np.random.negative_binomial(n = 1, p = 0.25, size=500))), infoFields="age")
    
        self.pop.setVirtualSplitter(sim.CombinedSplitter([
            sim.ProductSplitter([
                sim.SexSplitter(),
                sim.InfoSplitter(field = "age", cutoff = [1,3,6,10])])],
            vspMap = [[0,1], [2], [3], [4], [5,6,7,8], [9] ]))
    
        # Age groups: from 0 to 1 - cubs, from 1 to 3 - prereproductive, from 3 to 6 - reproductive class, from 6 to 10 - dominant
        self.pop.evolve(
            initOps=[
                sim.InitSex(),
                # random genotype
                sim.InitGenotype(freq=[0.01]*2 + [0.03]*2 + [0.23]*4),
                # assign an unique ID to everyone.
                sim.IdTagger(),
            ],
            # increase the age of everyone by 1 before mating.
            preOps=[sim.InfoExec('age += 1'),
                    sim.InfoExec("hc +=1 if 0 < hc < 3  else 0"), # Mother bear can't have cubs for two years after pregnancy
                    sim.Migrator(rate=[[self.cro_to_slo]],
                                 mode=sim.BY_PROPORTION,
                                 subPops=[(0, 0)],
                                 toSubPops=[1]), # reproductive males migrate from Cro to Slo
                    sim.Migrator(rate=[[self.slo_to_cro]],
                                 mode=sim.BY_PROPORTION,
                                 subPops=[(1, 0)],
                                 toSubPops=[0]),
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
                        fatherChooser=sim.PyParentsChooser(generator=self.bearFather),
                        motherChooser=sim.PyParentsChooser(generator=self.bearMother)
                    ),
                    generator=sim.OffspringGenerator(ops=[
                        sim.InfoExec("age = 0"),
                        sim.IdTagger(),
                        #sim.PedigreeTagger(),
                        sim.ParentsTagger(),
                        sim.MendelianGenoTransmitter()
                    ], numOffspring=(sim.UNIFORM_DISTRIBUTION, 1, 3))),
                sim.CloneMating(subPops=[(0,0), (0,1), (0,2), (0,4), (1,0), (1,1), (1,2), (1,4)], weight=-1),
    
            ], subPopSize=popmodel.demoModel),
            # number of individuals?
            postOps = [
                #sim.PyOperator(func=popmodel.NaturalMortality),
                sim.PyOperator(func = popmodel.CalcNe, param={"me":self.me, "Ne":self.Ne}, begin=int(0.2*self.generations)),
                sim.PyOperator(func = popmodel.CalcLDNe, param={"me":self.me, "x":self.x}, begin=int(0.2*self.generations)),
                sim.PyOperator(func=popmodel.cullCountry,param={"slo_cull": self.slo_cull, "cro_cull": self.cro_cull}),
                       ],
    
            gen = self.generations
        ) 