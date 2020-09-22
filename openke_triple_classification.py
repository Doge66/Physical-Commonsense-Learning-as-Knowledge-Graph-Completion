import openke
from openke.module.model import TransE, TransD, RESCAL, DistMult, ComplEx, SimplE

import joblib
import torch
import numpy as np
from collections import defaultdict
import argparse
import os
import sys
import timeit
import time

from data import (
    TASK_REV_MEDIUMHAND,
    TASK_LABELS,
)
import metrics
from utils import Task, openke_predict, get_entity_relationship_dicts

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='transe')
args = parser.parse_args()

ent_list, rel_list = get_entity_relationship_dicts()

if args.model=='transe':
    model = TransE(
        ent_tot = len(ent_list),
        rel_tot = len(rel_list),
        dim = 200, 
        p_norm = 1, 
        norm_flag = True)
elif args.model=='transd':
    model = TransD(
        ent_tot = len(ent_list),
        rel_tot = len(rel_list),
        dim_e = 200, 
        dim_r = 200, 
        p_norm = 1, 
        norm_flag = True)
elif args.model=='rescal':
    model = RESCAL(
        ent_tot = len(ent_list), 
	    rel_tot = len(rel_list),
	    dim = 50)
elif args.model=='distmult':
    model = DistMult(
        ent_tot = len(ent_list),
        rel_tot = len(rel_list),
        dim = 200)
elif args.model=='complex':
    model = ComplEx(
        ent_tot = len(ent_list),
        rel_tot = len(rel_list),
        dim = 200)
elif args.model=='simple':
    model = SimplE(
        ent_tot = len(ent_list),
        rel_tot = len(rel_list),
        dim = 200)

model = model.cpu()

start_time = timeit.default_timer()

task_names = ['situated-OP', 'situated-OA', 'situated-AP']

for task_name in task_names:
    print('{} task'.format(task_name))

    task = Task(TASK_REV_MEDIUMHAND[task_name])
    samples = task.get_test_examples()

    y_hat = []
    y = []
    names = []
    for sample in samples:
        names.append(sample.name)
        head, tail = sample.name.split('/')
        if task_name=='situated-OP':
            res = openke_predict(model, np.array(ent_list[head+'-o']), np.array(ent_list[tail+'-p']), np.array([[0],[1],[2]]), 0)
        elif task_name=='situated-OA':
            res = openke_predict(model, np.array(ent_list[head+'-o']), np.array(ent_list[tail+'-a']), np.array([[0],[1],[2]]), 1)
        elif task_name=='situated-AP':
            res = openke_predict(model, np.array(ent_list[head+'-a']), np.array(ent_list[tail+'-p']), np.array([[0],[1],[2]]), 2)
        y_hat.append(res)
        y.append(int(sample.label))  

    y = np.array(y)
    y_hat = np.array(y_hat)
    txt = metrics.report(y_hat, y, names, TASK_LABELS[TASK_REV_MEDIUMHAND[task_name]])
    print(txt)

stop_time = timeit.default_timer()
print('triple classification testing time: {}'.format((stop_time-start_time)))
    

