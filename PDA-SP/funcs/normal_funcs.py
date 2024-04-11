import random 
from typing import *
import numpy as np
import json
import pandas as pd
from pandas import DataFrame as df
import os


# import func_mine
import funcs.func_mine

class LoadData():
    def __init__(self) -> None:
        dt_fea_inf = pd.read_csv('data/df_fea_inf.csv')
        self.df_fea_inf = pd.DataFrame(dt_fea_inf)
        self.dt = funcs.func_mine.loadVariJson('data/preprocess/dt')
    def key_day(self, start:int, end:int)->List[str]:
        """返回windou_size天的训练数据需要用到的unique_key

        Args:
            start (int): _description_
            end (int): _description_

        Returns:
            List[str]: _description_
        """
        return list(set(self.df_fea_inf.loc[(self.df_fea_inf['dt']>=start) & (self.df_fea_inf['dt']<=end)]['unique_key']))
    def key_day_eva(self, dt_eva)->List[str]:
        """返回windou_size天的训练数据需要用到的unique_key

        Args:
            start (int): _description_
            end (int): _description_

        Returns:
            List[str]: _description_
        """
        return funcs.func_mine.loadVariJson(os.path.join('data/evaluation', '_'.join(['key_win', str(dt_eva)])))
    def cal_B(start:int, end:int)->float:
        """原本用于实时计算window_size大小的滑窗的base得分，后来我将base每天的得分提前算出，到时候累加即可。

        Args:
            start (int): _description_
            end (int): _description_

        Returns:
            float: _description_
        """
        pass

def isTerminal(xi, xi_old, threshold, grad):
    if abs(xi[0] - xi_old[0]) > threshold: return False
    for i in range(1, len(xi)):
        if grad[i] > 0:
            return False
    return True