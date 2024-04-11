import random 
from typing import *
import numpy as np
import json
import pandas as pd
from pandas import DataFrame as df
import ast
from funcs import func_mine as func_mine



class Oracle(object):
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
        if not isinstance(idx[0], list):
            idx = [idx]
        # return self.predict(np.stack(self.idx2x(unique_key, idx), axis=0)) # numpy
        return self.predict(self.idx2x(unique_key, idx)) # List


class Oracle_new(object):
    def __init__(self):
        self.w = 1.8636439
        self.b = 0.06237198
        self.key_idx_fea = func_mine.loadVariJson('data\preprocess\key_idx_fea_lst')
        self.key_idx_psco = func_mine.loadVariJson('data\preprocess\key_idx_psco')
    
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
        x, pscore = self.idx2x(unique_key, idx)
        return self.predict(x, pscore)

# Decision formulas for candidate sets to maximize tasks
def decision_score(xi:float, solution:List[List[int]], score:List[float])->float:
    if not isinstance(score, list) and not isinstance(score, np.ndarray):
        solution = [solution]
        score = [score]
    return [len(solution[i]) - xi * score[i] for i in range(len(score))]


# (l-knapsack constraint) Decision formula for candidate sets to maximize tasks
def decision_score_general(xi:List[float], solution:List[List[int]], score:List[float], cost:List[List[float]])->List[float]:
    if not isinstance(score, list) and not isinstance(score, np.ndarray):
        solution = [solution]
        score = [score]
        cost = [cost]
    dec_sco = []
    for i in range(len(score)):
        dec_sco.append(len(solution[i]) - xi[0] * score[i] + sum([xi[j]*cost[i][j-1] for j in range(1, len(xi))]))
    return dec_sco

# After fixing the solution, the objective function of xi (which may contain Re) is minimized to facilitate derivation
def loss_dual(xi, score, B)->float:
    return xi * (sum(score) - B)

def loss_grad(score, B):
    return sum(score) - B

def loss_general(xi, cost):
    return sum([xi[i]*cost[i] for i in range(len(xi))])

# ! Different forms of the problem, the specific analytic formula is also different
# def cal_alterxi(xi:List[float], i:int, min_idx:int, idx:int, cost_numerator:List[float], cost_denominator:List[float], obj:List[float]=None):
def cal_alterxi(xi:List[float], k:int, j:int, gloopt_minidx:int, min_idx:int, cost:List[List[List[float]]], obj:List[List[List[float]]]=None)->float:
    """_summary_

    Args:
        xi (List[float]): dual variables
        k (int): The index of the dual variable element that is currently being updated
        j (int): strategy size, ub_i-lb_i, j and gloopt_minidx have the same status, but j==gloopt_minidx does not occur
        gloopt_minidx (int): Optimal strategy index. Lines to be skipped (argmin decisionscore, min_idx[i][j])
        min_idx (int): A 2-dimensional array marking the subscript of Claimant i's optimal choice in each strategy size.
        cost (List[List[List[float]]]): cost is a 3-dimensional array, with layer 1 representing the strategy size, layer 2 representing the strategy subscript, and layer 3 representing the constraint subscript (score constraint or l-knapsack subscript). There's actually an outer layer and a layer that represents the Claimant.
        obj (List[List[List[float]]], optional)
    """
    if obj is None: obj = 1 
    numerator, denominator = obj, (cost[j][min_idx[j]][k]-cost[gloopt_minidx][min_idx[gloopt_minidx]][k])
    for _k in range(len(xi)):
        if _k == k:
            continue
        numerator += xi[_k]*(cost[j][min_idx[j]][_k]-cost[gloopt_minidx][min_idx[gloopt_minidx]][_k])
    return numerator / denominator
    
def find_opt_xi(alter_xi:List[float], xi:float, grad:float):
    if abs(grad) < 1e-5:
        print('Subgradient information is small and not updated')
        return xi
    gap, gapPos, gapNeg = float('inf'), float('inf'), float('inf')
    for alterxi_i in alter_xi:
        if grad > 0:
            if alterxi_i > xi and alterxi_i - xi < gap:
                gap = alterxi_i - xi
        else:
            if alterxi_i < xi and xi - alterxi_i < gap:
                gap = xi - alterxi_i
    if gap == float('inf'):
        if xi < 1e-5:
            print('If no feasible dual variable solution is found, and the dual variable is less than 1e-5, it will not be updated')
            return xi
        if grad > 0:
            return xi * 1.5
        else: 
            return xi * 0.5
    if grad > 0: return xi + gap
    else: return xi - gap
    
def find_opt_xi2(alter_xi:List[float], xi:float, grad:float):
    if abs(grad) < 1e-5:
        print('Subgradient information is small and not updated')
        return xi
    gap, gapPos, gapNeg = float('inf'), float('inf'), float('inf')
    for alterxi_i in alter_xi:
        gap = alterxi_i - xi
        if gap > 0 and gap < gapPos:
            gapPos = gap
        elif gap < 0 and gap > gapNeg:
            gapNeg = gap
    if (grad > 0 and gapPos != float('inf')):
        return xi + gapPos
    elif (grad < 0 and gapNeg != float('inf')):
        return xi + gapNeg
    elif gapPos != float('inf'):
        return xi + gapPos
    else:
        return xi + gapNeg
        
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

    def cal_sco(self, unique_key:str, idx:List[List[int]])->List[float]:
        """对外的接口

        Args:
            unique_key (str): id number
            idx (List[List[int]]):A two-dimensional query list that wraps multiple queries. A single query= strategy =List[int], representing the subscript of the candidate set for selecting the hospital.

        Returns:
            List[float]: Calculated cost/resource overhead
        """
        if not isinstance(idx[0], list):
            idx = [idx]
        return self.idx2cost(unique_key, idx)
    
