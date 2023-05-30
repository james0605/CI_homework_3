import numpy as np
import RBFN
import random
import time
import math
# -----hyper parameter-----
NUMBER = 3
K = 2
LR_ID = 0.1
LR_G = 0.1
# -----PSO Algo-----
class PSO(object):
    def __init__(self, train_x, train_y, Number = NUMBER, k = K):
        # init data set
        self.train_x = train_x
        self.train_y = train_y
        
        # init every RBFN's X(weight) to rand
        self.RBFNs_x = {i:np.random.randn(k + 1) for i in range(Number)}
        # print(self.RBFNs_x[0])
        
        # init every RBFN's V(move) to 0
        self.RBFNs_v = {i:np.zeros(k+1) for i in range(Number)}

        # init personal best Position(P_id) and group best Position(P_g)
        self.P_g = {"X":np.zeros(k+1), "fitness":0}
        self.P_id = {i:{"X":np.zeros(k+1), "fitness":0} for i in range(Number)}
        # print("P_g = {}".format(self.P_g))
        # print("P_id = {}".format(self.P_id))

        # create a RBFNet with centers
        self.RBFN = RBFN.RBFNet(K)
        self.RBFN.get_center(train_x)

    # -----update P_g-----
    def find_best_group_position(self):
        fitness = np.array([1/self.RMSE_loss(value) for key, value in self.RBFNs_x.items()])
        best_arg = fitness.argmin()
        best_fitness = fitness[best_arg]
        return best_arg, best_fitness

    def update_P_g(self):
        best_arg, best_fitness = self.find_best_group_position()
        self.P_g["X"] = self.RBFNs_x[best_arg]
        self.P_g["fitness"] = best_fitness

    # -----update P_id-----
    def cal_fitness(self, weight):
        fitness = 1/self.RMSE_loss(weight)
        return fitness
    
    # def find_best_personal_position(self, index):
    #     fitness = self.cal_fitness(self.RBFNs_x[index])
    #     if self.P_id[index]["fitness"] < fitness:
    #         self.P_id[index]["fitness"] = fitness
    #         self.P_id[index]["X"] = self.RBFNs_x[index]
    
    def update_V(self):
        for i in self.RBFNs_v.keys():
            self.RBFNs_v[i] = self.RBFNs_v[i] + LR_ID*(self.P_id[i]["X"] - self.RBFNs_x[i]) + LR_G*(self.P_g - self.RBFNs_x[i])

    def update_X(self):
        for i in self.RBFNs_x.keys():
            if self.cal_fitness(self.RBFNs_x[i]) < self.cal_fitness(self.RBFNs_x[i]+self.RBFNs_v[i]):
                self.RBFNs_x[i] += self.RBFNs_v[i]

        pass        


    def cal_V(self):

        pass

    
    def predict(self, states, weight):
        # init weight and test data
        self.RBFN.w = weight
        # states = [states]
        result = []
        print(states)
        # test 
        for state in states:
            result.append(self.RBFN.predict(state))
        result = np.array(result)
        print("result :{}".format(result))
        return result
    
    def RMSE_loss(self, Genetic):
        # calculate loss
        predict = self.predict(self.train_x, Genetic)
        loss = math.sqrt(np.sum((predict - self.train_y)**2)/len(self.train_x))
        return loss
    
# -----test-----
if __name__ == "__main__":
    
    train_data = np.array(RBFN.getTrain4d())
    X = train_data[:, :-1]
    Y = train_data[:, -1]
    p = PSO( train_x=X,train_y= Y, Number = NUMBER, k = K)
    best_arg, best_fitness = p.find_best_group_position()
    print(best_arg, best_fitness)

    