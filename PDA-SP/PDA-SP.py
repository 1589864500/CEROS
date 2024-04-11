import torch
from torch.autograd import Variable
from torch import nn 
import torch.nn.functional as F


import matplotlib.pyplot as plt
import numpy as np
from typing import *
import random
import math
import os
import random
import tqdm


import funcs.func_mine as tool
import algs.greedy_general
import algs.guessK_greedy_general
import funcs.normal_funcs
import funcs.key_funcs_notorch


# ! Main Code Logic
## Read in the dataset, initialize parameters
## Begin training
###     for until the change in dual variables \xi is less than 1e-05 or the maximum number of iterations is reached:
###          With dual variables \xi fixed, solve for a batch of recommended investigation strategies for Claimants. Claimants are from a fixed time span (e.g., one week), updated in a sliding window fashion.
###          Update the \xi vector by the submodualr optimization algorithm with segmentation point (vector length = number of constraints during dualization)


# ! Read in the dataset, initialize parameters
## The `unique_key` holds data with ['claim_report_no', 'dt'] as the primary key. Claimants on different dates are treated as distinct samples.
unique_key = tool.loadVariJson('data/preprocess/unique_key') 
## Initialize parameters 1：Parameters that do not need to be tuned
stop_thres = epsilon = 1e-05
constraint_num = 1
B = tool.loadVariPickle('data/preprocess/409base_f')
C = tool.loadVariPickle('data/preprocess/409base_lknapsack')
key_hosnum = tool.loadVariJson('data/preprocess/key_hosnum')
key_hosdom = tool.loadVariJson('data/preprocess/key_hosdom')
lod_dt = funcs.normal_funcs.LoadData()
dt = tool.loadVariJson('data/preprocess/dt')
win_siz = 7 # The width of the sliding window. Here, it's assumed that the sample distribution for the next day follows the distribution of the past week's historical data.
start_idx = 14
start = dt[start_idx]
c = 1 # Scaling factor, to maintain consistency with other comparison methods.
oracle = funcs.key_funcs_notorch.Oracle_new()
l = 3 # The number of knapsack constraints.
## Initialize parameters 2：Parameters that can be adjusted
method = 'naivegreedy'
# method = 'guessKgreedy'
max_iter = 100
# eva_num = 1 # Controls the number of days for evaluation. It's recommended to use 7 (days) in industrial settings.
eva_num = 1 #
show_round = 1

def train(start:int, end:int, end_next:int, B:float, C:List[float], xi) -> float:
    """Update dual variables by the submodualr optimization algorithm with segmentation point 

    Args:
        start (int): Time slice at the start of the training data set
        end (int): Time slice at the end of the training data set
        end_next (int): Time slice of validation data
        B (List[float]): The amount of l-knapsack constrained resources
        C (List[float]): The amount of l-knapsack constrained resources
        xi (List[float]): Dual variable (initialization)

    Returns:
        List[float]: Dual variable (after convergence)
    """
    fname_lostra = 'data/results/plot_log' # The result storage directory.
    fname_xitra = 'data/results/xi_log'
    if os.path.exists(fname_lostra):
        plot_log = tool.loadVariJson(fname_lostra)
    else:
        plot_log = {} # Result logs, List[List[float]].
    plot_log[str(end_next)] = []
    fname_xi = 'data/results/xi'
    if os.path.exists(fname_xi):
        XI = tool.loadVariJson(fname_xi)
    else:
        XI = {}
    key_win = lod_dt.key_day(start, end) # Read the data for the number of sliding windows (unique_key, used for xi training).
    curv_xi:List[List[float]] = []
    for iter in tqdm.tqdm(range(1, max_iter+1)):
        if iter % show_round == 0:
            print('Every {} rounds, the new xi is {}.'.format(show_round, xi))
        xi_old = xi[:]
        curv_xi.append(xi_old)
        ## With \xi fixed, solve for a batch of recommended investigation strategies for Claimants. Claimants are from a fixed time span (e.g., one week), updated in a sliding window fashion.
        solution, score, cost = [], [], [] # Stores the iteration results of the submodular optimization algorithm for each round when given xi (not yet converged).
        min_idx, solution_general, cost_general, gloopt_minidx = [], [], [], [] # Stores the intermediate processes of each round of iterations by the submodular optimization algorithm when given xi (not yet converged).
        for _, key in enumerate(key_win):
        # for _, key in enumerate(key_win[:3]): # ! DEBUG
            dom = key_hosdom[key]
            if method == 'naivegreedy':
                min_idx_i, solution_general_i, cost_general_i, gloopt_minidx_i = algs.greedy_general.naive_greedy(xi, key, dom, key_hosnum[key], oracle) # 3-dimensional array data structure.
            elif method == 'guessKgreedy':
                print('TODO')
                exit()
                # solution_i, score_i, decision_score_i = algs.guessK_greedy_general.guessK_greedy(xi, key, dom, key_hosnum[key], oracle, k=min(dom[-1], 3))
            # For dual variables updating.
            solution_general.append(solution_general_i)
            cost_general.append(cost_general_i)
            min_idx.append(min_idx_i)
            gloopt_minidx.append(gloopt_minidx_i)
            # Display the results, optimization iteration outcomes.
            solution.append(solution_general_i[gloopt_minidx_i][min_idx_i[gloopt_minidx_i]])
            score.append(cost_general_i[gloopt_minidx_i][min_idx_i[gloopt_minidx_i]][0])
            cost.append(cost_general_i[gloopt_minidx_i][min_idx_i[gloopt_minidx_i]][1:])
        
        # Maximization Task
        obj = sum([len(s) for s in solution]) # Cost
        sco = B - sum(score) # Subgradient information
        cst = [sum(costi[j] for costi in cost) - C[j] for j in range(len(cost[0]))] # Subgradient information
        cst_expand = [sco] + cst # Subgradient information
        print('Cost：{}, Subgradient：{}'.format(obj, cst_expand))
        loss = funcs.key_funcs_notorch.loss_general(xi, cst_expand)
        plot_log[str(end_next)].append([loss, obj]+cst_expand) # Display results, constraint violation outcomes.
        
        ## Update the \xi vector by the submodualr optimization algorithm with segmentation point (vector length = number of constraints during dualization)
        # Step1: argmin decisionscore, min_idx[i][j]
        # Step2: for j from 1 to ub, compare the best of round J-1 (min_idx[i][j]) with all the choices of round j (find the intersection).
        # Step3: In combination with the sub-gradient information, the intersection that is closest to the current dual variable is retained
        for k in range(len(xi)): # k represents the dual component of renewal
            alter_xi = []
            # Step2
            for i in range(len(min_idx)): # |D|times, i represents Claimant
                # For each Claimant i, we can get 1 possible xi
                alter_xi_i = []
                for j in range(0, len(solution_general[i])): # Size of the strategies
                    if j == gloopt_minidx[i]:
                        continue
                    alter_xi_i.append(funcs.key_funcs_notorch.cal_alterxi(xi, k, j, gloopt_minidx[i], min_idx[i], cost_general[i]))
                alter_xi += alter_xi_i
            # Step3
            # Warning: If a feasible solution for the dual variables is not found, do not update (adjustment with random numbers is also possible).
            xi[k] = funcs.key_funcs_notorch.find_opt_xi2(alter_xi, xi[k], cst_expand[k]) 
        
        # Legitimacy check for dual variables.
        for i in range(len(xi)):
            if xi[i] < 0 - epsilon:
                xi[i] = 5 * random.random()
        
        # Stopping condition (convergence)
        if funcs.normal_funcs.isTerminal(xi, xi_old, stop_thres, cst_expand):
            print('The number of iterations for early termination is {}.'.format(iter))
            break
        
    # Results display
    color = tool.getColor()
    label = ['xi'+str(i) for i in range(len(curv_xi[0]))]
    for i in range(len(curv_xi[0])):
        plt.plot(range(len(curv_xi)), [xi_i[i] for xi_i in curv_xi], color[i], label=label[i])
    plt.legend()
    fname_save_xi = os.path.join('data/results', str(end_next)+'_xi')
    plt.savefig(fname_save_xi)
    plt.clf()
    label = ['total_loss', 'objective', 'score'] + ['knapsack'+str(i) for i in range(1, len(cst_expand))]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(len(label)):
        ax.plot(range(len(plot_log[str(end_next)])), [loss_i[i] for loss_i in plot_log[str(end_next)]], color[i], label=label[i])
    # ax2 = ax.twinx()
    # ax2.plot(range(len(plot_log[str(end_next)])), [los[1] for los in plot_log[str(end_next)]], 'g', label='score')
    plt.legend()
    fname_save_los = os.path.join('data/results', str(end_next)+'_cost_score')
    plt.savefig(fname_save_los)
    plt.clf()
    
    # Storage
    XI[str(end_next)] = xi
    tool.dumpVariJson(plot_log, fname_lostra)
    tool.dumpVariJson(curv_xi, fname_xitra)
    tool.dumpVariJson(XI, fname_xi)
    return xi

# ! Begin training
for start_i in range(start_idx, start_idx+eva_num):
    start = dt[start_i]
    for i, dti in enumerate(dt): # end dt is calculated by start dt
        if dti == start:
            B_train = sum([B[dt[i + j]] for j in range(win_siz)])
            end = dt[i+win_siz-1]
            end_next = dt[i+win_siz]
    C_train = C[start+win_siz]
    if l == 3:
        xi_init_path = 'data/results/groundtruth/xi_general-l=3' 
    elif l == 5:
        xi_init_path = 'data/results/groundtruth/xi_general-l=5'
    else:
        print('data/results/groundtruth/xi_general-l=? not exist!')
        exit()
    if not os.path.exists(xi_init_path):
        tool.dumpVariJson({}, xi_init_path)
    xi_init = tool.loadVariJson(xi_init_path)
    if str(end) in xi_init:
        xi  = xi_init[str(end)] 
    else: 
        xi = [random.random()*10 for _ in range(constraint_num)]
    print('-------Start training dt={}-{}---------'.format(start, end))
    print('Initialize the dual variables to:{}'.format(xi))
    xi = train(start, end, end_next, B_train*c, C_train, xi)
    xi_init[str(end_next)] = xi
    tool.dumpVariJson(xi_init, xi_init_path)