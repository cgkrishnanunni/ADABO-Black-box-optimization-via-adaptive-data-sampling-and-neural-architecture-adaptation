"""
Cuckoo Search Optimization (CSO) used to minimize the surrogate model.

Population members live in normalized coordinates. Fitness is evaluated
through the neural surrogate (see ``optimal_csa.fitness_1``), not the
true black-box — that expensive evaluation happens only after CSO finishes.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import gamma
from joblib import Parallel, delayed


class CSO:
    """Cuckoo Search with Lévy flights over a bounded box."""

    def __init__(self, fitness, data, weights, biases, hyperp, initialization, P, n, pa=0.25, beta=1.5, bound=None,
                plot=False, min=True, verbose=False, Tmax=300, seed=0):

        self.data = data
        self.fitness = fitness
        self.hyperp = hyperp
        self.weights = weights
        self.biases = biases
        self.P = P          # population size
        self.n = n          # design dimension
        self.Tmax = Tmax    # generations
        self.pa = pa        # discovery / abandon probability
        self.beta = beta    # Lévy flight exponent
        self.bound = bound
        self.plot = plot
        self.min = min
        self.verbose = verbose

        # Fixed seed for Lévy flights / discovery randomness
        np.random.seed(seed)

        # Initialize population of nests
        self.X = []

        #if bound is not None:
            #for (U, L) in bound:
                #x = (U-L)*np.random.rand(P,) + L 
                #self.X.append(x)
            #self.X = np.array(self.X).T
        #else:
            #self.X = np.random.randn(P,n)

        self.X=initialization

        # Initialize best solution and fitness cache
        fitness_values = self.evaluate_population(self.X)
        if self.min:
            best_idx = np.argmin(fitness_values)
        else:
            best_idx = np.argmax(fitness_values)
        self.best = self.X[best_idx].copy()
        self.best_fitness = fitness_values[best_idx]

    def evaluate_population(self, X):
        # Parallel evaluation of fitness for population X (list/array of shape (P,n))
        return self.fitness(X,self.hyperp,self.weights,self.biases)

    def update_position_1(self):

        num = gamma(1 + self.beta) * np.sin(np.pi * self.beta / 2)
        den = gamma((1 + self.beta) / 2) * self.beta * (2 ** ((self.beta - 1) / 2))
        sigma_u = (num / den) ** (1 / self.beta)
        sigma_v = 1
        u = np.random.normal(0, sigma_u, self.n)
        v = np.random.normal(0, sigma_v, self.n)
        S = u / (np.abs(v) ** (1 / self.beta))

        Xnew = self.X.copy()

        # Move each particle according to Levy flight
        for i in range(self.P):
            step = np.random.randn(self.n) * 0.01 * S * (Xnew[i, :] - self.best)
            Xnew[i, :] += step

        # Clip new positions within bounds before fitness evaluation
        if self.bound is not None:
            for i in range(self.n):
                L, U = self.bound[i]
                Xnew[:, i] = np.clip(Xnew[:, i], L, U)

        # Evaluate fitness in parallel for new solutions
        fitness_new = self.evaluate_population(Xnew)
        fitness_old = self.evaluate_population(self.X)

        # Update particles and global best based on fitness comparison
        for i in range(self.P):
            if self.min:
                if fitness_new[i] < fitness_old[i]:
                    self.X[i, :] = Xnew[i, :]
                    if fitness_new[i] < self.best_fitness:
                        self.best = Xnew[i, :].copy()
                        self.best_fitness = fitness_new[i]
            else:
                if fitness_new[i] > fitness_old[i]:
                    self.X[i, :] = Xnew[i, :]
                    if fitness_new[i] > self.best_fitness:
                        self.best = Xnew[i, :].copy()
                        self.best_fitness = fitness_new[i]

    def update_position_2(self):

        Xnew = self.X.copy()
        Xold = self.X.copy()

        for i in range(self.P):
            d1, d2 = np.random.randint(0, self.P, 2)  # Use population size, not fixed 5
            for j in range(self.n):
                r = np.random.rand()
                if r < self.pa:
                    Xnew[i, j] += np.random.rand() * (Xold[d1, j] - Xold[d2, j])

        # Clip new positions within bounds before fitness evaluation
        if self.bound is not None:
            for i in range(self.n):
                L, U = self.bound[i]
                Xnew[:, i] = np.clip(Xnew[:, i], L, U)

        # Evaluate fitness in parallel
        fitness_new = self.evaluate_population(Xnew)
        fitness_old = self.evaluate_population(self.X)

        # Update particles and global best
        for i in range(self.P):
            if self.min:
                if fitness_new[i] < fitness_old[i]:
                    self.X[i, :] = Xnew[i, :]
                    if fitness_new[i] < self.best_fitness:
                        self.best = Xnew[i, :].copy()
                        self.best_fitness = fitness_new[i]
            else:
                if fitness_new[i] > fitness_old[i]:
                    self.X[i, :] = Xnew[i, :]
                    if fitness_new[i] > self.best_fitness:
                        self.best = Xnew[i, :].copy()
                        self.best_fitness = fitness_new[i]

    def clip_X(self):
        if self.bound is not None:
            for i in range(self.n):
                L, U = self.bound[i]
                self.X[:, i] = np.clip(self.X[:, i], L, U)


    def execute(self):



        #self.fitness_time, self.time = [], []
       # store_f=[]
        store_sol=[]
        #prob_sol=[]
        for t in range(self.Tmax):

 

            
                
            self.update_position_1()
            self.clip_X()
            self.update_position_2()
            self.clip_X()
                


            

            



        
            

            #bt=self.fitness(self.best,self.hyperp)

            best_solution=self.best
            #print('Iteration:  ',t,'| best global fitness (cost):',bt)
           # store_f.append(bt)
            store_sol.append(self.best)


            #np.savetxt('best_solution_cuc.txt', store_sol)


            YY=np.zeros((np.shape(self.data)[0],np.shape(self.data)[1]))

            for i in range(0,np.shape(self.data)[0]):
                YY[i,:]=best_solution-self.data[i,:]

            if np.min(np.linalg.norm(YY,axis=1))<0.01:
                print(f"breaking iteration of CSA: {t}")
                break
           # np.savetxt('best_fun.txt', store_f)
            

   
        if t==0:
            L, U = self.bound[0]

            random_vector = np.random.randn(self.n)

    # Calculate the magnitude (Euclidean norm) of the vector
            magnitude = np.linalg.norm(random_vector)

    # Normalize the vector to unit length
            unit_vector = random_vector / magnitude

            store_sol[len(store_sol)-2]=np.clip(best_solution+list(0.01*unit_vector),L,U)

        return store_sol[len(store_sol)-2]