from copy import deepcopy
import random
import sys
from collections import namedtuple, OrderedDict
import pandas as pd

N = 100
POPULATION_SIZE = N         
OFFSPRING_SIZE = N*2        
NUM_GENERATIONS = N*10
ARTIFICIAL_MUTATIONS= 35000 
MAX_STEADY=10
MAX_EXTINCTIONS=10   
Individual = namedtuple("Individual", ["genome", "fitness"])
TOURNAMENT_SIZE =int(N/4)
GENETIC_OPERATOR_RANDOMNESS = 0.3
MUTATION_THRESHOLD = 0.1
CROSSOVER_THRESHOLD = 0.5

#PROBLEM PARAMETERS
INITIAL_STAMINA=None
MAX_STAMINA=None
N_TURNS=None
N_DEMONS=None
RIGHE_FILE=None
#DATA
best_individual=(None, sys.float_info.min)
list_of_lists=[]

#FABIO
def take_data(nome_file):
    global list_of_lists
    global INITIAL_STAMINA
    global MAX_STAMINA
    global N_TURNS
    global N_DEMONS
    f = open(nome_file, "r")
    intestazione = f.readline().split()
    INITIAL_STAMINA= int(intestazione[0])
    MAX_STAMINA= int(intestazione[1])
    N_TURNS= int(intestazione[2])
    N_DEMONS = int(intestazione[3])
    for line in f:
        line=line.split()
        line=[int(i) for i in line]
        list_of_lists.append(line)
    f.close()

def print_data_output(result):
    s= pd.Series(result)
    s.to_csv("output",index=False, header=False)
    
#LEO

def compute_fitness(genome):
    global list_of_lists
    global INITIAL_STAMINA
    global MAX_STAMINA
    global N_TURNS
    global N_DEMONS
    fitness=0
    gene_index=0
    staminas=[0]*N_TURNS
    fragments=[0]*N_TURNS
    stamina=INITIAL_STAMINA
    for turn in range(N_TURNS):
        stamina+=staminas[turn]
        if stamina>MAX_STAMINA:
            stamina=MAX_STAMINA
        demon=list_of_lists[genome[gene_index]]
        if turn==0 and demon[0]>stamina:
            print("INITIAL STAMINA NOT ENOUGH TO START")
            return fitness
        if demon[0]<=stamina:
            stamina-=demon[0]
            gene_index+=1
            if turn+demon[1]<N_TURNS:
                staminas[turn+demon[1]]+=demon[2]
            for i in range(0,demon[3]):
                if turn+i>=N_TURNS:
                    break
                fragments[turn+i]+=demon[4+i]
        fitness+=fragments[turn]
    return fitness
        
        
        
###########FLAVIO##################

"""parent selection"""
def tournament(population, tournament_size=TOURNAMENT_SIZE):
    global TOURNAMENT_SIZE          
    return max(random.choices(population, k=TOURNAMENT_SIZE), key=lambda i: i.fitness) 
    
"""generate our initial population"""

def init_population():
    global INITIAL_STAMINA
    global MAX_STAMINA
    global N_TURNS
    global N_DEMONS
    global POPULATION_SIZE
    population = []
    for _ in range(POPULATION_SIZE):
        genome = []
        for i in range(N_DEMONS):
            genome.append(i)
        random.shuffle(genome)
        genome=genome[:N_TURNS]
        population.append(Individual(genome, compute_fitness(genome)))
        

    return population

"""mutation"""

def mutation(genome):
    global INITIAL_STAMINA
    global MAX_STAMINA
    global N_TURNS
    global N_DEMONS 
    new_genome = deepcopy(genome)
    max_mutations=N_DEMONS//100
    if max_mutations==0:
        max_mutations=1
    n_mutations=random.randint(1,max_mutations)
    
    for i in range( n_mutations):
        pos_1 = random.randint(0,N_TURNS-1)
        val_2= random.randint(0,N_DEMONS-1)
        try:
            pos_2=new_genome.index(val_2)
            val_1 = new_genome[pos_1]
            new_genome[pos_1] = val_2
            new_genome[pos_2] = val_1
        except ValueError:
            new_genome[pos_1] = val_2

    return new_genome

"""crossover"""
def cross_over(genome_1, genome_2):
    global INITIAL_STAMINA
    global MAX_STAMINA
    global N_TURNS
    global N_DEMONS
    global CROSSOVER_THRESHOLD
    new_genome = []
    set_tot_val = {i for i in range(N_DEMONS)}
    for i in range(0, N_TURNS):
        if (random.random() > CROSSOVER_THRESHOLD):
            new_genome.append(genome_1[i])
        else:
            new_genome.append(genome_2[i])

    new_genome_plus=list(OrderedDict.fromkeys(new_genome))
    residual = list(set_tot_val - set(new_genome_plus))
    cut=N_TURNS-len(new_genome_plus)
    residual=residual[:cut]
    new_genome_plus=new_genome_plus+residual

    return new_genome_plus


"""evolution"""
def evolution(population):
    global list_of_lists
    global best_individual
    global INITIAL_STAMINA
    global MAX_STAMINA
    global N_TURNS
    global N_DEMONS
    global MAX_EXTINCTIONS
    global NUM_GENERATIONS
    global MAX_STEADY
    global OFFSPRING_SIZE
    global POPULATION_SIZE
    global GENETIC_OPERATOR_RANDOMNESS
    check_steady = 0
    check_extinctions=0
    generation=0
    print("BEGIN EVOLUTION")
    while(check_extinctions<=MAX_EXTINCTIONS and generation<NUM_GENERATIONS):
        generation+=1
        print("generation: ",generation)
        offspring = list()
        for i in range(OFFSPRING_SIZE):
            if random.random() < GENETIC_OPERATOR_RANDOMNESS:                         
                p = tournament(population)                  
                o = mutation(p.genome)                    
            else:                                          
                p1 = tournament(population)                 
                p2 = tournament(population)
                o = cross_over(p1.genome, p2.genome)            
            f = compute_fitness(o)                                                          
            offspring.append(Individual(o, f))                 
        population += offspring      

        #unique population
        unique_population = []
        unique_genomes = []
        for individual in population:
            if individual.genome not in unique_genomes:
                unique_genomes.append(individual.genome)
                unique_population.append(individual)

        population = sorted(unique_population, key=lambda i: i[1], reverse=True)[:POPULATION_SIZE]

        #check actual best individual
        actual_best_individual=Individual(population[0][0],population[0][1])

        if actual_best_individual[1] > best_individual[1]:
            best_individual=actual_best_individual
            check_steady = 0
        else:
            check_steady+= 1

        if check_steady == MAX_STEADY:
            check_extinctions+=1
            check_steady = 0
            new_population = init_population()
            final_population = []
            for i in range(len(population)):
                if random.random() > 0.15: #70% new population
                    final_population.append(new_population[i])
                else:
                    final_population.append(population[i]) #30% old population
            population=final_population
    if generation<NUM_GENERATIONS:
        check_extinctions-=1       
    print("TOT GENERATIONS: ", generation, "/", NUM_GENERATIONS)
    print("TOT EXTINCTIONS: " , check_extinctions , "/" , MAX_EXTINCTIONS)

def artificial_evolution():
    global list_of_lists
    global best_individual
    global ARTIFICIAL_MUTATIONS
    artificial_population=[]
    artificial_population.append(best_individual)
    gen=0
    gen_a=0
    found=False 
    print("BEGIN ARTIFICIAL EVOLUTION")
    for ind in artificial_population:
        gen+=1
        print("artificial generation: ",gen)
        for a in range(ARTIFICIAL_MUTATIONS):
            o=mutation(ind.genome)
            f = compute_fitness(o)
            frankenstein=Individual(o,f)
            if best_individual[1] < frankenstein[1]:
                found=True 
                best_individual=Individual(frankenstein[0], frankenstein[1])
                gen_a=a+1
        if found==True: 
            artificial_population.append(best_individual) 
            found=False 
    print("ARTIFICIAL GENERATIONS: ", gen, "+", gen_a)

def complete_output():
    global list_of_lists
    global best_individual
    global N_TURNS
    global N_DEMONS
    if N_TURNS<N_DEMONS:
        demons=[]
        for i in range(N_DEMONS):
            demons.append(i)
        remaining_demons=list(set(demons)-set(best_individual[0]))
        best_genome=best_individual[0]+remaining_demons
        best_individual=Individual(best_genome, best_individual[1])


if __name__ == '__main__':
    take_data("05-androids-armageddon.txt")#01-the-cloud-abyss.txt
    #print(list_of_lists)
    population=init_population()
    evolution(population)
    print("EVOLUTION SCORE: " , best_individual[1])
    population=None
    artificial_evolution()
    #print(best_individual[0])
    complete_output()
    print("FINAL SCORE: " , best_individual[1])
    print_data_output(best_individual[0])
           



    
    

    