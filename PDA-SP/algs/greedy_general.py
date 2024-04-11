import numpy as np
import math
from typing import *
import random
import copy


import sys
import os
sys.path.append(os.getcwd())
print(sys.path)
import funcs.key_funcs_notorch

# Submodular function optimization, and return (List) solution
## Outputs the (List) solution, along with the score in float form.
def naive_greedy(xi:List[float], key:str, dom:List[int], hos_num:int, oracle)->List[List[float]]:
    """The simple greedy algorithm is used to find the optimal strategy corresponding to each strategy size, and finally the optimal strategy size is selected by the decision formula.

    Args:
        xi (List[float]): Dual variables, len= number of constraints
        key (str): Claim id
        dom (List[int]): Mark the number of hospitals that can be deployed (base constraint, describing the range of strategy sizes)
        hos_num (int): The number of candidate hospitals must not be less than the upper limit of the number of hospitals that can be transferred (Strategy Set E)

    Returns:
        List[List[float]]: In order to reduce double calculations, greedy computes all corresponding strategies
        return1: List[int]
        return2: List[List[List[float]]], Layer 1 represents the strategy size, layer 2 represents the subscript of a strategy of the same size, and layer 3 represents the specific elements of a strategy
        return3: List[List[List[float]]]
        return4: int
    """
    is_selected = [0] * hos_num # Complete set of candidate hospitals E, different candidate sets for different Claimants.
    solution= [] # The solution starts with an empty set
    opt_decision_score = float('inf') # Minimization task
    lknapsack = funcs.key_funcs_notorch.l_knapsack(len(xi))
    opt_minidx, opt_solutiongeneral, opt_costgeneral, gloopt_minidx = [], [], [], -1
    for i in range(dom[-1]): #The maximum number of dom[-1] hospitals is adjusted, and the treated dom[-1] must not be larger than the size of the candidate hospital set |E|.
        alter_new:List[List[int]] = [solution + [hos_idx] for hos_idx in range(hos_num) if is_selected[hos_idx] == 0] # Si
        alter_score_new:List[float] = oracle.cal_sco(key, alter_new) # F(Si)
        alter_cost_new:List[List[float]] = lknapsack.cal_sco(key, alter_new) # F(Si)
        alter_decision_score_new:List[np.ndarray] = funcs.key_funcs_notorch.decision_score_general(xi, alter_new, alter_score_new, alter_cost_new) # np.ndarray.shape=(1,1)
        min_idx = np.argmin(alter_decision_score_new) # best index, min task
        # Record the results of each iteration
        opt_minidx.append(min_idx)
        opt_solutiongeneral.append(alter_new)
        opt_costgeneral_i = [[alter_score_new[i]]+alter_cost_new[i] for i in range(len(alter_new))] # All resource overhead
        opt_costgeneral.append(opt_costgeneral_i) 
        # Update global optimal
        solution, decision_score = alter_new[min_idx], alter_decision_score_new[min_idx]
        is_selected[solution[-1]] = 1 # Removes the selected element
        if decision_score < opt_decision_score:
            gloopt_minidx, opt_decision_score = i, decision_score
    return opt_minidx, opt_solutiongeneral, opt_costgeneral, gloopt_minidx
        