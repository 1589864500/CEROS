import random 
from typing import *
import numpy as np
import json
import pandas as pd
from pandas import DataFrame as df
import ast
from funcs import func_mine as func_mine


from funcs import func_mine as func_mine

# Oracle, 具体使用时会采用SQL查表

class Oracle(object):
    """For convenience, oracle supports batch access of the type

    Args:
        unique_key (str): _description_
        solution (List[int]): _description_

    Returns:
        _type_: _description_
    """
    def __init__(self, weight_file='funcs\dsf_weights.txt'):
        with open(weight_file, 'r') as f:
            data = json.load(f)
        self.key_idx_fea = func_mine.loadVariJson('data\preprocess\key_idx_fea_lst')
        self.w1 = np.array(data['weight_1'])
        self.w2 = np.array(data['weight_2'])
        

    def sigmoid(self, x):
        return 1 / (1+np.exp(-x))

    def predict(self, x):
        # x: List[nd.array], size=[batch, 256]
        # print(x.shape)
        ## A potential problem with neural network-based scoring models is that there are points less than -1, which will result in np.log1p reporting an illegal Warning value
        if np.min(np.matmul(x, self.w1)) < -1:
            pass # DEBUG
        h = np.log1p(np.matmul(x, self.w1))
        h = np.matmul(h, self.w2)
        prob = 2*self.sigmoid(h) - 1
        return prob
    
    def idx2x(self, unique_key:str, idx:List[List[int]])->List[List[float]]:
        def fea_mer(f_list:List[List[float]])-> List[float]:
            """Combine multiple sets of features from a hospital portfolio into a single feature

            Args:
                f_list (List[List[float]]): _description_

            Returns:
                List[float]: _description_
            """
            f = [np.stack(f_list_i, axis=1) for f_list_i in f_list]
            return [1-np.prod(1-f_i, axis=1) for f_i in f]
        
        f_list_str = []
        for idx_i in idx:
            f_list_str.append([self.key_idx_fea[unique_key][idx_i_j] for idx_i_j in idx_i])
        f_list = []
        for f_list_i in f_list_str:
            f_list.append(f_list_i)
        return fea_mer(f_list)
    
    def cal_sco(self, unique_key:str, idx:List[List[int]])->List[float]:
        """score calculation

        Args:
            unique_key (List[str]): _description_
            idx (List[List[int]]): _description_

        Returns:
            List[float]: _description_
        """
        if not isinstance(idx[0], list):
            idx = [idx]
        # return self.predict(np.stack(self.idx2x(unique_key, idx), axis=0)) # numpy
        return self.predict(self.idx2x(unique_key, idx)) # List

class Oracle_new(object):
    def __init__(self):
        self.w = 1.8636439
        self.b = 0.06237198
        self.key_idx_fea = func_mine.loadVariJson('data/preprocess/key_idx_fea_lst')
        self.key_idx_psco = func_mine.loadVariJson('data/preprocess/key_idx_psco')
    
    def sigmoid(self, x):
        return 1 / (1+np.exp(-x))

    def predict(self, x, pscore):
        # x: [batch, k, 256]
        # pscore: [batch, k]
        if len(x.shape) == 2:
            x = x[np.newaxis]
        if len(pscore.shape) == 1:
            pscore = pscore[np.newaxis]

        k = pscore.shape[1]
        kernel = np.matmul(x, np.transpose(x, [0,2,1]))
        kernel *= pscore[:,np.newaxis] * pscore[:,:,np.newaxis]
        kernel += np.expand_dims(np.eye(k), axis=0)

        logdet = np.log(np.linalg.det(kernel))
        logit = logdet * self.w + self.b
        prob = 2*self.sigmoid(logit) - 1
        return prob
    
    def idx2x(self, unique_key:str, idx:List[List[int]])->np.ndarray: # List[List[float]]
        f_lst, ps_lst = [], []
        for idx_i in idx:
            f_lst.append([self.key_idx_fea[unique_key][idx_ij] for idx_ij in idx_i])
            ps_lst.append([self.key_idx_psco[unique_key][idx_ij] for idx_ij in idx_i])
        return np.array(f_lst), np.array(ps_lst)
    
    def cal_sco(self, unique_key:str, idx:List[List[int]])->List[float]:
        if not isinstance(idx[0], list):
            idx = [idx]
        f_lst, ps_lst = self.idx2x(unique_key, idx)
        return self.predict(f_lst, ps_lst)
        
   
# Decision formulas for candidate sets to maximize tasks
def decision_score(xi:float, solution:List[List[int]], score:List[float])->List[float]:
    if not isinstance(score, list) and not isinstance(score, np.ndarray):
        solution = [solution]
        score = [score]
    return [len(solution[i]) - xi.detach().numpy() * score[i] for i in range(len(score))]

# (l-knapsack constraint) Decision formula for candidate sets to maximize tasks
def decision_score_general(xi:np.ndarray, solution:List[List[int]], score:List[float], cost:List[List[float]])->List[float]:
    if not isinstance(score, list) and not isinstance(score, np.ndarray):
        solution = [solution]
        score = [score]
        cost = [cost]
    dec_sco = []
    xi = xi.detach().numpy()
    for i in range(len(score)):
        dec_sco.append(len(solution[i]) - xi[0] * score[i] + sum([xi[j]*cost[i][j-1] for j in range(1, len(xi))]))
    return dec_sco

# After fixing the solution, the objective function of xi (which may contain Re) is minimized to facilitate derivation
def loss_dual(xi, score, B)->float:
    return xi * (sum(score) - B)

def loss_general(xi, cost):
    return sum([-1*xi[i]*cost[i] for i in range(len(xi))])

class l_knapsack:
    def __init__(self, l:int) -> None:
        csv_fname = 'data/df_fea_inf_new.csv'
        csv_data = pd.read_csv(csv_fname)
        self.df_fea_inf = pd.DataFrame(csv_data)
        if l == 5:
            cost_path = 'data/preprocess/l_knapsack-l=5'
        elif l == 3:
            cost_path = 'data/preprocess/l_knapsack-l=3'
        else:
            print("data/preprocess/l_knapsack not exist!")
            exit()
        cost = func_mine.loadVariJson(cost_path, showLog=False)
        self.cost = list(cost.values()) # The element of the list is the Hash table, hosid->cost

    def idx2cost(self, unique_key:str, idx:List[List[int]])->np.ndarray: # List[List[float]]
        key = self.df_fea_inf.loc[self.df_fea_inf['unique_key'] == unique_key]
        cost = []
        for idxi in idx:
            cost.append([])
            hosid = [key.loc[key['hos_idx']==idxij]['fence_id'].values[0] for idxij in idxi]
            for i in range(len(self.cost)):
                cost[-1].append(sum([self.cost[i][hosidj] for hosidj in hosid]))
        return cost
        
    def cal_sco(self, unique_key:str, idx:List[List[int]])->List[float]:
        if not isinstance(idx[0], list):
            idx = [idx]
        return self.idx2cost(unique_key, idx)
    