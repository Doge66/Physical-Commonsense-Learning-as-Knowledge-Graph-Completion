import joblib
import torch
import numpy as np
from collections import defaultdict
import argparse
import os

from load_data import Data
from data import (
    TASK_REV_MEDIUMHAND,
    TASK_LABELS,
)
import metrics
from utils import Task, get_entity_sets

def eval_tc(saved_model_path):
    batch_size = 128
    cuda = True if torch.cuda.is_available() else False
    add_constraint = True
    add_dropout_bn = False

    dic = joblib.load(os.path.join(saved_model_path, 'dic.pkl'))
    model = torch.load(os.path.join(saved_model_path, 'model.pt'), map_location=torch.device('cpu'))

    if cuda:
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

    res = []
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

            if (th+1) % batch_size==0 or th+batch_size>=len(test_samples):
                input1 = torch.tensor(input1)
                input2 = torch.tensor(input2)
                if cuda:
                    input1 = input1.cuda()
                    input2 = input2.cuda()

                outputs = model.forward(input1, input2)
                if add_dropout_bn==True:
                    predictions = outputs
                else:
                    predictions = outputs[0]
                    W = outputs[1]
                    E = outputs[2]
                    R = outputs[3]

                if cuda:
                    predictions = predictions.cpu()
                
                for k, e12 in enumerate(zip(input1, input2)):
                    e1 = e12[0].cpu().item()
                    e2 = e12[1].cpu().item()
                    r = e12_type(entity_reverse_idxs[e1], entity_reverse_idxs[e2])
                    if add_constraint==True:
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
        acc,micro_f1,macro_f11,macro_f12 = metrics.report_more4cv(y_hat, y, names, TASK_LABELS[TASK_REV_MEDIUMHAND[task_name]])
        res.append((acc,micro_f1,macro_f11,macro_f12))

    return res


def eval_lp(saved_model_path, data_path):
    cuda = True
    add_constraint = True
    batch_size = 512
  
    model = torch.load(os.path.join(saved_model_path, 'model.pt'), map_location=torch.device('cpu'))
    if cuda:
        model = model.cuda()    

    d = Data(data_dir=data_path, reverse=True, add_constraint=add_constraint)

    entity_idxs = {d.entities[i]:i for i in range(len(d.entities))}
    relation_idxs = {d.relations[i]:i for i in range(len(d.relations))}
    idx2entity = {v:k for k,v in entity_idxs.items()}
    idx2relation = {v:k for k,v in relation_idxs.items()}

    def _get_data_idxs(data):
        data_idxs = [(entity_idxs[data[i][0]], relation_idxs[data[i][1]], \
                        entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs

    def _get_er_vocab(data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def _get_batch(er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx+batch_size]
        targets = np.zeros((len(batch), len(d.entities)))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets)
        if cuda:
            targets = targets.cuda()
        return np.array(batch), targets

    hits = []
    ranks = []
    for i in range(10):
        hits.append([])

    test_data_idxs = _get_data_idxs(d.test_data)
    er_vocab = _get_er_vocab(_get_data_idxs(d.data))

    print("Number of data points: %d" % len(test_data_idxs))
    
    for i in range(0, len(test_data_idxs), batch_size):
        data_batch, _ = _get_batch(er_vocab, test_data_idxs, i)
        e1_idx = torch.tensor(data_batch[:,0])
        r_idx = torch.tensor(data_batch[:,1])
        e2_idx = torch.tensor(data_batch[:,2])
        if cuda:
            e1_idx = e1_idx.cuda()
            r_idx = r_idx.cuda()
            e2_idx = e2_idx.cuda()
        outputs = model.forward_lp(e1_idx, r_idx)
        predictions = outputs[0]
        W = outputs[1]
        E = outputs[2]
        R = outputs[3]

        for j in range(data_batch.shape[0]):
            filt = er_vocab[(data_batch[j][0], data_batch[j][1])]
            target_value = predictions[j,e2_idx[j]].item()
            predictions[j, filt] = 0.0
            predictions[j, e2_idx[j]] = target_value

        sort_values, sort_idxs = torch.sort(predictions, dim=1, descending=True)

        sort_idxs = sort_idxs.cpu().numpy()
        for j in range(data_batch.shape[0]):
            rank = np.where(sort_idxs[j]==e2_idx[j].item())[0][0]
            ranks.append(rank+1)

            for hits_level in range(10):
                if rank <= hits_level:
                    hits[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)

    print('Hits @10: {0}'.format(np.mean(hits[9])))
    print('Hits @3: {0}'.format(np.mean(hits[2])))
    print('Hits @1: {0}'.format(np.mean(hits[0])))
    print('Mean rank: {0}'.format(np.mean(ranks)))
    print('Mean reciprocal rank: {0}'.format(np.mean(1./np.array(ranks))))

    return np.mean(1./np.array(ranks)), np.mean(hits[9]), np.mean(hits[2]), np.mean(hits[0])