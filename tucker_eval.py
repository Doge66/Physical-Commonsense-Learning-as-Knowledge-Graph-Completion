import joblib
import torch
import numpy as np
from collections import defaultdict
import argparse
import os
import timeit

from data import (
    TASK_REV_MEDIUMHAND,
    TASK_LABELS,
)
import metrics
from utils import Task, get_entity_sets


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--cuda", type=bool, default=True if torch.cuda.is_available() else False, nargs="?",
                help="Whether to use cuda (GPU) or not (CPU).")
parser.add_argument("--saved_model_path", type=str, default='tucker_models')
parser.add_argument("--add_constraint", action='store_true')
parser.add_argument("--add_dropout_bn", action='store_true')
args = parser.parse_args()

dic = joblib.load(os.path.join(args.saved_model_path, 'dic.pkl'))
model = torch.load(os.path.join(args.saved_model_path, 'model.pt'), map_location=torch.device('cpu'))

if args.cuda:
    model = model.cuda()
entity_idxs = dic[0]
relation_idxs = dic[1]
entity_reverse_idxs = {i:name for name, i in entity_idxs.items()}
relation_reverse_idxs = {i:name for name, i in relation_idxs.items()}
model.eval()

# gather objects, properties, and affordances
task_names = ['situated-OP', 'situated-OA', 'situated-AP']
task_mapping = defaultdict()

for name in task_names:
    task_mapping[name] = Task(TASK_REV_MEDIUMHAND[name])

objects, properties, affordances = get_entity_sets(task_mapping)

def e12_type(e1, e2):
    if (e1 in objects and e2 in properties):
        return 'situated-OP'
    elif (e1 in objects and e2 in affordances):
        return 'situated-OA'
    elif (e1 in affordances and e2 in properties):
        return 'situated-AP'
    else:
        return 'nothing'

start_time = timeit.default_timer()
for task_name, task in task_mapping.items():
    print('{} task'.format(task_name))

    y_hat = []
    y = []
    names = []
    input1 = []
    input2 = []
    test_samples = task.get_test_examples()

    for th, sample in enumerate(test_samples):
        names.append(sample.name)

        e12 = sample.name.split('/')
        input1.append(entity_idxs[e12[0]])
        input2.append(entity_idxs[e12[1]])

        if (th+1) % args.batch_size==0 or th+args.batch_size>=len(test_samples):
            input1 = torch.tensor(input1)
            input2 = torch.tensor(input2)
            if args.cuda:
                input1 = input1.cuda()
                input2 = input2.cuda()

            outputs = model.forward(input1, input2)
            if args.add_dropout_bn==True:
                predictions = outputs
            else:
                predictions = outputs[0]
                W = outputs[1]
                E = outputs[2]
                R = outputs[3]

            if args.cuda:
                predictions = predictions.cpu()
            
            for k, e12 in enumerate(zip(input1, input2)):
                e1 = e12[0].cpu().item()
                e2 = e12[1].cpu().item()
                r = e12_type(entity_reverse_idxs[e1], entity_reverse_idxs[e2])
                if args.add_constraint==True:
                    if r is not 'nothing':
                        v1 = predictions[k, relation_idxs[r]]
                        v2 = predictions[k, relation_idxs['NOT-'+r]]
                        if v1>v2:
                            y_hat.append(1)
                        else:
                            y_hat.append(0)
                    else:
                        y_hat.append(0)
                else:
                    if r is not 'nothing':
                        v = predictions[k, relation_idxs[r]].item()
                        if v>=0.5:
                            y_hat.append(1)
                        else:
                            y_hat.append(0)
                    else:
                        y_hat.append(0)

            input1 = []
            input2 = []
        y.append(int(sample.label))  
    y = np.array(y)
    y_hat = np.array(y_hat)
    txt = metrics.report(y_hat, y, names, TASK_LABELS[TASK_REV_MEDIUMHAND[task_name]])
    print(txt)

stop_time = timeit.default_timer()
print('testing time: {}'.format(stop_time-start_time))

