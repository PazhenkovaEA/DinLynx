from scipy.stats import bernoulli
import random as random
def NaturalMortality(pop, param):
    all_inds = [x for x in pop.individuals(0)]
    sampling_m = random.choices(all_inds, k=int(param["Mortality"] * len(all_inds))) #Adult mortality 0.19-0.29
    out_ids = [x.ind_id for x in sampling_m]
    pop.removeIndividuals(IDs=out_ids, idField="ind_id")
    return True

def InbreedingLoad(pop, param):
    ids = [ind.ind_id for ind in pop.individuals() if ind.IBD != 0]
    not_survival = []
    for ind in ids:
        F = pop.indByID(ind).IBD
        survival_probability = 2.71828 ** (- float(param["B"]) * F) #eq 4 from Nietlisbach et al 2019 (doi: 10.1111/eva.12713)
        survival = int(bernoulli.rvs(size=1, p=survival_probability))
        if survival == 0:
            not_survival.append(ind)
    pop.removeIndividuals(IDs=not_survival, idField="ind_id")
    return True

def IBD(pop):
    ids = [ind.ind_id for ind in pop.individuals() if ind.IBD==0]
    for ind in ids:
        if (pop.indByID(ind).father_id == 0) or (pop.indByID(ind).mother_id == 0):
            F = 0
        else:
            offspring = len(set(pop.indByID(ind).lineage()))
            try:
                mother = len(set(pop.indByID(pop.indByID(ind).mother_id).lineage()))
            except:
                continue
            try:
                father = len(set(pop.indByID(pop.indByID(ind).father_id).lineage()))
            except:
                continue
            F = (1 - (offspring/(mother+father)))*0.5
        pop.indByID(ind).setInfo(F, 'IBD')
    return True