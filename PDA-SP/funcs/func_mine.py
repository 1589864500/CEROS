import datetime
import functools


import numpy as np
from functools import reduce
from typing import *


import pickle
import json
import os


def getTime(format=None):
    if format is None:
        return datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    else:
        return datetime.datetime.now().strftime(format)
# NOTE Pickle
def dumpVariPickle(vari: Any, path:str=None, name:str=None, showLog=True):
    if path is None:
        path = os.getcwd()
        path = os.path.join(path, name)
    elif name is not None:
        path = os.path.join(path, name)
    with open(path, 'wb') as f:
        pickle.dump(vari, f)
        f.close()
    if showLog: print('Write: ', path)
def loadVariPickle(path:str, showLog=True) ->Any:
    # path.key = vari_name, path.value = vari
    if showLog: print('Read: ', path)
    with open(path, 'rb') as f:
        para = pickle.load(f)
        f.close()
        return para
# NOTE Json

def dumpVariJson(vari: Any, path:str=None, name:str=None, indent=4, showLog=True):
    if path is None:
        if not os.path.isabs(name):
            path = os.getcwd()
            path = os.path.join(path, name)
        else:
            path = name
    elif name is not None:
        if isinstance(path, list):
            path_i = os.getcwd()
            for dir in path:
                path_i = os.path.join(path_i, dir)
            path = path_i
        path = os.path.join(path, name)
    with open(path, 'w') as f:
        json.dump(vari, f, indent=indent)
        f.close()
    if showLog: print('Write: ', path)
def loadVariJson(path:str=None, name:str=None, showLog=True) ->Any:
    # path.key = vari_name, path.value = vari
    if path is None:
        path = name
    if isinstance(path, list):
        path_i = os.getcwd()
        for dir in path:
            path_i = os.path.join(path_i, dir)
        path = path_i
    if showLog: print('Read: ', path)
    with open(path, 'r') as f:
        data = f.read()
        para = json.loads(data)
        f.close()
        return para
def creatDir(dir:str)->None:
    if not os.path.exists(dir):
        os.mkdir(dir)


def getColor()->List[str]:
    return ['b','g','r','c','m','y','k','w']