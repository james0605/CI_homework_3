import numpy as np
import RBFN
import random
import time
import math
'''
self.Muta_Fra = 0.3
self.Cross_Fra = 0.5
self.REPET_fra= 0.2
G_num = 50
k:15
'''

class GeneticOpt(object):
    def __init__(self, G_num = 5, RBFN_K = 10, dump = False, load = False) -> None:
        self.Genetics = []
        self.pool = []
        # self.traindata = [dist for dist in RBFN.getTrain4d()]
        self.traindata = np.array(RBFN.getTrain4d())
        self.RBFN = RBFN.RBFNet(RBFN_K)
        self.winner = np.random.randn(RBFN_K + 1) * np.random.randn(RBFN_K + 1)
        self.dump = dump
        self.is_done = False
        self.Muta_Fra = 0
        self.Cross_Fra = 0.6
        self.REPET_fra= 0.1
        self.G_NUM = G_num
        self.RBFN_K = RBFN_K
        if load:
            center = np.load('./best/centers.npy')
            w = np.load('./best/weight.npy')
            self.RBFN_K = len(w)
            self.RBFN.centers = center
            self.RBFN.w = w
            self.winner = w
            
            print("centers number {}".format(self.RBFN.centers.shape))
            print("weight number {}".format(self.RBFN.w.shape))
        else:
            for i in range(G_num):
                weight = np.random.randn(RBFN_K + 1) * np.random.randn(RBFN_K + 1)
                self.Genetics.append(weight)
        print(self)

    def check_Genetic_w(self):
        print("Genetic Pool :")
        for i, Genertic in enumerate(self.Genetics):
            print("Genertic {} weight:{}".format(i, Genertic))

    # 複製
    def reproduction(self):
        start = time.time()
        # fitness, avg_fitness = self.cal_fitness_v2()

        # 線性調整
        # min_fit = min(fitness)
        # max_fit = max(fitness)
        # fitness = np.array(fitness)
        # fitness = fitness - min_fit
        # d = max_fit - min_fit
        # if d == 0:
        #     d=1
        # print(fitness)
        # fitness /= d
        # print(type(fitness))
        # avg_fitness = np.mean(fitness)
        # print("avg_fitness :{}".format(avg_fitness))
        pool = []

        fitness = np.array([self.RMSE_loss(Genetic) for Genetic in self.Genetics])
        # print("fitness :{}".format(fitness))
        sorted_fitness = np.argsort(fitness)
        print("BEST Genetic RMSE loss :{}".format(self.RMSE_loss(self.Genetics[sorted_fitness[0]])))
        WINER_NUM = int(self.G_NUM * self.REPET_fra)
        # print("sorted_fitness arg before repr:{}".format(sorted_fitness))
        sorted_fitness[-WINER_NUM:] = sorted_fitness[0]
        # print("sorted_fitness arg after repr:{}".format(sorted_fitness))
        for index in sorted_fitness:
            pool.append(self.Genetics[index])
            
        # print("POOL NUM : {}".format(len(pool)))
        # print("sorted fitness arg:{}".format(sorted_fitness))

        if len(pool) == 0:
            pool.append(self.Genetics[0])
        end = time.time()
        # print("reproduction run time :{}".format(end - start))
        return pool
    # 交配
    def crossover(self, Genetic1, Genetic2, sigma = 0.5):
        start = time.time()
        temp = Genetic1
        Genetic1 = Genetic1 + sigma*(Genetic1 - Genetic2)
        Genetic2 = Genetic2 - sigma*(temp - Genetic2)
        end = time.time()
        # print("crossover run time :{}".format(end - start))
        return Genetic1, Genetic2
    
    # 突變
    def mutation(self, Genetic, s = 1):
        start = time.time()
        rand = np.random.uniform(low=-1.0, high=1.0)
        # Genetic += s * rand
        Genetic += s * 0.3
        # Genetic *= s 
        end = time.time()
        # print("mutation run time :{}".format(end - start))
        return Genetic
    # 訓練
    def fit(self, epoch):
        for i in range(epoch):
            # old_Genetics = self.Genetics.copy()
            # old_avg = self.cal_avg_fitness()

            # print("There is {} Genetics in Pool".format(len(self.Genetics)))
            # if len(set(self.Genetics)) == 1:
            #     print("Train Done")
            #     self.winner = self.Genetics[0]
            #     break
            print("epoch:{}/{}".format(i, epoch))
            if self.dump:
                print("{:-^50s}".format("Origin"))
                self.check_Genetic_w()
            end = time.time()
            # 複製

            start = time.time()
            self.Genetics = self.reproduction()
            if self.dump:
                print("{:-^50s}".format("After Reproduction"))
                self.check_Genetic_w()
            end = time.time()
            # print("Repuoduction run time in fit :{}".format(end - start))
            # 交配
            start = time.time()
            rand_list = np.arange(0, len(self.Genetics))
            rand_list = list(rand_list)
            for i in range(int(self.G_NUM * self.Cross_Fra // 2)):
                rand_selet = random.sample(rand_list, k=2)
                self.Genetics[rand_selet[0]], self.Genetics[rand_selet[1]] =  self.crossover(self.Genetics[rand_selet[0]], self.Genetics[rand_selet[1]])
                if self.dump:
                    print("{:-^50s}".format("After Crossover"))
                    print("Choose {} and {}".format(rand_selet[0], rand_selet[1]))
                    self.check_Genetic_w()
            end = time.time()
            # print("Crossover run time in fit :{}".format(end - start))
            # 突變
            start = time.time()
            for i in range(int(self.G_NUM * self.Muta_Fra)):
                muta_rand = random.choice(rand_list)
                self.Genetics[muta_rand] = self.mutation(self.Genetics[muta_rand])
                if self.dump:
                    print("{:-^50s}".format("After Mutation"))
                    print("Choose {}".format(muta_rand))
                    self.check_Genetic_w()
            end = time.time()
            # print("Mutation run time in fit :{}".format(end - start))
            # print("loss:{}".format(self.cal_loss()))
            # print("1 / old avg :{}".format(1/old_avg))
            # print("1 / new avg :{}".format(1/self.cal_avg_fitness()))
            # if 1/old_avg < 1/self.cal_avg_fitness():
            #     self.Genetics = old_Genetics.copy()
            #     print("Choose Ex-epoch pool")
            

        # if len(set(self.Genetics)) != 1:
        fitness = np.array([self.RMSE_loss(Genetic=Genetic) for Genetic in self.Genetics])
        self.winner = self.Genetics[fitness.argmin()]
        self.RBFN.w = self.winner
        np.save("RBFN_Center", self.RBFN.centers)
        np.save("RBFN_Weight", self.RBFN.w)
    
    def RMSE_loss(self, Genetic):
        data, y = self.traindata[:, :-1], self.traindata[:, -1]
        predict = self.predict(data, Genetic)
        # print("predict shape:{}".format(predict.shape))
        loss = math.sqrt(np.sum((predict - y)**2)/len(self.traindata))
        # print("RMSE run time :{}".format(end - start))
        return loss


    def cal_fitness(self, Genetic):
        data, y = self.traindata[:, :-1], self.traindata[:, -1]
        predict = self.predict([data], Genetic)
        loss = 1/(np.sum((predict - y)**2))
        return loss
    

    def get_w(self, i):
        return self.Genetics[i]
    
    def predict(self, states, weight):
        self.RBFN.w = weight
        states = [states]
        result = []
        for state in states:
            result.append(self.RBFN.predict(state))
        result = np.array(result)
        # print("result shape:{}".format(result.shape))
        print("result :{}".format(result))
        return result

if __name__ == "__main__":
    ga = GeneticOpt(20, 58, dump = False, load=True)    
    # print(ga.RMSE_loss(ga.Genetics[0]))
    print(np.shape(ga.Genetics))
    # print(ga.Genetics[0].predict([9.7355, 10.9379, 18.5740]))
    # ga.fit(epoch=1)
    w  = ga.RBFN.w
    ga.predict([11.5458, 31.8026, 11.1769], w) # 40
    # ga.predict([9.7355, 10.9379, 18.5740], winner) # -40