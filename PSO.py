import numpy as np
import RBFN
import random
import time
import math
# -----hyper parameter-----
# NUMBER = 5
# K = 4
# LR_ID = 0.3
# LR_G = 0.3
# EPOCH = 10
# -----PSO Algo-----
class PSOpt(object):
    def __init__(self, train_x, train_y,LR_ID = 0.3, LR_G = 0.3, Number = 10, k = 5, Load = False):
        # init data set
        self.train_x = train_x
        self.train_y = train_y
        self.LR_ID = LR_ID
        self.LR_G = LR_G
        self.Number = Number
        # create a RBFNet with centers
        self.RBFN = RBFN.RBFNet(k)
        self.RBFN.get_center(train_x)
        
        # init every RBFN's X(weight) to rand
        self.RBFNs_x = {i:np.random.randn(k + 1) for i in range(self.Number)}
        # print(self.RBFNs_x[0])
        
        # init every RBFN's V(move) to 0
        self.RBFNs_v = {i:np.zeros(k+1) for i in range(self.Number)}

        # init personal best Position(P_id) and group best Position(P_g)
        max_index = np.array([self.cal_fitness(self.RBFNs_x[i]) for i in range(self.Number)]).argmax()
        self.P_g = {"X":self.RBFNs_x[max_index], "fitness":self.cal_fitness(self.RBFNs_x[max_index])}
        self.P_id = {i:{"X":self.RBFNs_x[i], "fitness":self.cal_fitness(self.RBFNs_x[i])} for i in range(self.Number)}
        # print("P_g = {}".format(self.P_g))
        # print("P_id = {}".format(self.P_id))

        # -----LOAD-----
        if Load:
            self.RBFN.centers = np.load('weight/RBFN_Center.npy')
            self.RBFN.w = np.load('weight/RBFN_Weight.npy')
            self.RBFN.stds = np.load('weight/RBFN_stds.npy')

    def cal_fitness(self, weight):
        fitness = 1/self.RMSE_loss(weight)
        return fitness

    # -----update P_g-----
    def find_best_group_position(self):
        fitness = np.array([self.cal_fitness(value) for key, value in self.RBFNs_x.items()])
        best_arg = fitness.argmax()
        best_fitness = fitness[best_arg]
        return best_arg, best_fitness

    def update_P_g(self):
        print("updating P_g")
        best_arg, best_fitness = self.find_best_group_position()
        self.P_g["X"] = self.RBFNs_x[best_arg]
        self.P_g["fitness"] = best_fitness

    # -----update P_id-----
    def update_P_id_and_x(self):
        print("updating P_id and x...")
        for i in range(self.Number):
            new_fitness = self.cal_fitness(self.RBFNs_x[i]+self.RBFNs_v[i])
            if new_fitness > self.P_id[i]["fitness"]:
                self.P_id[i]["X"] = (self.RBFNs_x[i] + self.RBFNs_v[i])
                self.P_id[i]["fitness"] = new_fitness
                self.RBFNs_x[i] = self.P_id[i]["X"]
    
    def cal_V(self):
        print("cal V.....")
        for i in self.RBFNs_v.keys():
            self.RBFNs_v[i] = self.RBFNs_v[i] + self.LR_ID*(self.P_id[i]["X"] - self.RBFNs_x[i]) + self.LR_G*(self.P_g["X"] - self.RBFNs_x[i])
        return self.RBFNs_v
    # -----perdict-----
    def predict(self, states, weight):
        # init weight and test data
        self.RBFN.w = weight

        result = []
        # test 
        for state in states:
            result.append(self.RBFN.predict(state))
        result = np.array(result)
        return result
    
    def RMSE_loss(self, Genetic):
        # calculate loss
        predict = self.predict(self.train_x, Genetic)
        loss = math.sqrt(np.sum((predict - self.train_y)**2)/len(self.train_x))
        return loss
    # -----train-----
    def fit(self, epoch):
        for i in range(epoch):
            print("epoch : {}".format(i))

            self.RBFNs_v = self.cal_V()
            self.update_P_id_and_x()
            self.update_P_g()
            print(self.RMSE_loss(self.P_g["X"]))
            
        self.RBFN.w = self.P_g["X"]
        np.save("RBFN_Center", self.RBFN.centers)
        np.save("RBFN_Weight", self.RBFN.w)
        np.save("RBFN_stds", self.RBFN.stds)

# -----test-----
if __name__ == "__main__":
    
    train_data = np.array(RBFN.getTrain4d())
    X = train_data[:, :-1]
    Y = train_data[:, -1]
    p = PSOpt( train_x=X, train_y= Y, Number = NUMBER, k = K)
    # p.predict()
    
    p.fit(EPOCH)
    # best_arg, best_fitness = p.find_best_group_position()
    # print(best_arg, best_fitness)
    # 12.0000 16.9706 08.4853 040.000
    p.predict([12.0000, 16.9706, 08.4853], p.P_g["X"])

    