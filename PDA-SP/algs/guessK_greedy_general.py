## Considering the efficiency factor, bit encoding is used for state compression and cache table is used for acceleration


import numpy as np
import math
from typing import *
import random
import copy
from functools import *


import sys
import os
sys.path.append(os.getcwd())
print(sys.path)
import funcs.key_funcs_notorch



## guessK and list2binary are both part of the guess-k algorithm.
def guessK(hos_maxlength:int, k:int=3)->List[List[int]]:
    res = []
    def backtrack(resi:List[int], k:int):
        if k == 0: 
            res.append(resi.copy())
            return 
        start = 0 if len(resi) == 0 else resi[-1] + 1
        for i in range(start, hos_maxlength):
            backtrack(resi+[i], k-1)
    backtrack([], k)
    return res
def list2binary(l:List[List[int]])->List[int]:
    res = []
    for li in l:
        resi = 0
        for pos in li:
            resi |= (1 << pos)
        res.append(resi)
    return res

@lru_cache(None) # ! Cache decorator
def binary2list(b:int)->List[int]:
    res = []
    temp, pos = 1, 0
    while temp < b:
        if temp & b:
            res.append(pos)
        temp <<= 1
        pos += 1
    return res

# The solution representation of the greedy search is binary (binary for cache acceleration)
def naive_greedy(xi, key:str, k:List[int], hos_num:int, oracle, sol:int, table:Dict[int, int]): # The strategy starts with enumerate solution[item]
    opt_score, opt_decision_score = 0, float('inf') # Maximization task
    for i in range(1, k+1): # Represents the number of optional items left
        alter_new = []
        for j in range(hos_num):
            sol_new = sol | (1<<j)
            # if not ((1<<j) & sol) and sol_new not in table:
            if not ((1<<j) & sol):
                alter_new.append(sol_new)
                table[sol_new] = 1
        if len(alter_new) == 0: continue
        alter_score_new:List[float] = oracle.cal_sco(key, [binary2list(alter) for alter in alter_new])
        alter_decision_score_new:List[np.ndarray] = funcs.key_funcs_notorch.decision_score(xi, [binary2list(alter) for alter in alter_new], alter_score_new) # np.ndarray.shape=(1,1)
        min_idx = np.argmin(alter_decision_score_new)
        if alter_decision_score_new[min_idx] < opt_decision_score:
            sol, opt_score, opt_decision_score = alter_new[min_idx], alter_score_new[min_idx], alter_decision_score_new[min_idx]
    return binary2list(sol), opt_score, opt_decision_score, table

# The main difference from the naive_greedy function is the difference in the return value
def naive_greedy_alter(key:str, k:List[int], hos_num:int, oracle, sol:int, table:Dict[int, int]): # The strategy starts with enumerate solution[item]
    sol_len = len(binary2list(sol))
    alter = {sol_len + added:[[], 0] for added in range(1, k+1)}
    for i in range(1, k+1): # Represents the number of optional items left
        alter_new = []
        for j in range(hos_num):
            sol_new = sol | (1<<j)
            if not ((1<<j) & sol) and sol_new not in table:
            # if not ((1<<j) & sol):
                alter_new.append(sol_new)
                table[sol_new] = 1
        if len(alter_new) == 0: continue
        alter_score_new:List[float] = oracle.cal_sco(key, [binary2list(item) for item in alter_new])
        max_idx = np.argmax(alter_score_new)
        alter[sol_len + i] = [binary2list(alter_new[max_idx]), alter_score_new[max_idx]]
        sol = alter_new[max_idx]
    return alter, table

# ! Main Code Logic
## Outputs the (List) solution, along with the score in float form.
##  Use enumerate to compute alter_opt for <=k (since the dual submodular functions are not monotony, these alter_opt need to participate in the online decision process)
##  Then call enumerate(k=k) greedy times, and each time greedy returns a set of alter_opt(List[List[int]]) for >k scenarios.
##  Select alter_opt in the scenario >k, and use the decision function to select the optimal object
def guessK_greedy(xi, key:List[str], dom:List[int], hos_num:int, oracle, k:int=3):
    table = {}
    ## The optimal value of alter_opt in scenarios with enumerate <=k
    opt_dec_sco = float('inf')
    for i in range(1, k+1): # Complete the exhaustion of k=[1,k]
        guessk:List[List[int]] = guessK(hos_num, i)
        alter_score_new:List[float] = oracle.cal_sco(key, guessk)
        alter_decision_score_new:List[np.ndarray] = funcs.key_funcs_notorch.decision_score(xi, guessk, alter_score_new) # np.ndarray.shape=(1,1)
        min_idx = np.argmin(alter_decision_score_new)
        if alter_decision_score_new[min_idx] < opt_dec_sco:
            opt_sol, opt_sco, opt_dec_sco = guessk[min_idx], alter_score_new[min_idx], alter_decision_score_new[min_idx]
    ## Compute enumerate(k=k) times greedy
    sol_enu = list2binary(guessk) # By chance, the guessk of the last for loop is used, corresponding to the guessk in the k=k scenario.
    for sol in sol_enu:
        alt_opt_sol, alt_opt_sco, alt_opt_dec_sco, table = naive_greedy(xi, key, dom[-1]-k, hos_num, oracle, sol, table) # Since k is taken in advance, the number of choices left -k
        if alt_opt_dec_sco < opt_dec_sco:
            opt_sol, opt_sco, opt_dec_sco = alt_opt_sol, alt_opt_sco, alt_opt_dec_sco
    return opt_sol, opt_sco, opt_dec_sco
        