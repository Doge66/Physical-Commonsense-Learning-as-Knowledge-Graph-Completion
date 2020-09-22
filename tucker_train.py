from load_data import Data
import numpy as np
import torch
import time
import os
from collections import defaultdict
from model import TuckERNoDropoutBN
from model_dropout_bn import TuckER
from torch.optim.lr_scheduler import ExponentialLR
import argparse
import joblib
import torch.nn as nn
import timeit

from data import TASK_REV_MEDIUMHAND
from utils import Task, get_entity_sets
   
class Experiment:

    def __init__(self, learning_rate=0.0005, ent_vec_dim=200, rel_vec_dim=200, 
                 num_iterations=500, batch_size=128, decay_rate=0., 
                 cuda=False,
                 input_dropout=0.3, hidden_dropout1=0.4, hidden_dropout2=0.5,                 
                 label_smoothing=0.,
                 saved_model_path='.',
                 add_constraint=True,
                 add_dropout_bn=False,
                 do_link_prediction=False,
                 reg=0.1):
        self.learning_rate = learning_rate
        self.ent_vec_dim = ent_vec_dim
        self.rel_vec_dim = rel_vec_dim
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.label_smoothing = label_smoothing
        self.cuda = cuda
        self.add_constraint = add_constraint
        self.add_dropout_bn = add_dropout_bn
        self.do_link_prediction = do_link_prediction
        self.saved_model_path = saved_model_path
        self.reg = reg
        self.kwargs = {"input_dropout": input_dropout, "hidden_dropout1": hidden_dropout1,
                       "hidden_dropout2": hidden_dropout2}
        
    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]], \
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs
    
    def get_er_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            if self.do_link_prediction==True:
                er_vocab[(triple[0], triple[1])].append(triple[2])
            else:
                er_vocab[(triple[0], triple[2])].append(triple[1])
        return er_vocab

    def get_batch(self, er_vocab, er_vocab_pairs, idx):
        batch = er_vocab_pairs[idx:idx+self.batch_size]
        if self.do_link_prediction==True:
            targets = np.zeros((len(batch), len(d.entities)))
        else:
            targets = np.zeros((len(batch), len(d.relations)))
        for idx, pair in enumerate(batch):
            targets[idx, er_vocab[pair]] = 1.
        targets = torch.FloatTensor(targets)
        if self.cuda:
            targets = targets.cuda()
        return np.array(batch), targets

    def evaluate_link_prediction(self, model, data):
        start_time = timeit.default_timer()
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        test_data_idxs = self.get_data_idxs(data)
        er_vocab = self.get_er_vocab(self.get_data_idxs(d.data))

        print("Number of data points: %d" % len(test_data_idxs))
        
        for i in range(0, len(test_data_idxs), self.batch_size):
            data_batch, _ = self.get_batch(er_vocab, test_data_idxs, i)
            e1_idx = torch.tensor(data_batch[:,0])
            r_idx = torch.tensor(data_batch[:,1])
            e2_idx = torch.tensor(data_batch[:,2])
            if self.cuda:
                e1_idx = e1_idx.cuda()
                r_idx = r_idx.cuda()
                e2_idx = e2_idx.cuda()
            outputs = model.forward_lp(e1_idx, r_idx)
            if self.add_dropout_bn==True:
                predictions = outputs
            else:
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
            
        stop_time = timeit.default_timer()
        print('testing time: {}'.format(stop_time-start_time))

    def train(self):
        print("Training the model...")
        self.entity_idxs = {d.entities[i]:i for i in range(len(d.entities))}
        self.relation_idxs = {d.relations[i]:i for i in range(len(d.relations))}
        self.idx2entity = {v:k for k,v in self.entity_idxs.items()}
        self.idx2relation = {v:k for k,v in self.relation_idxs.items()}

        if self.add_constraint==True:
            # constrain types
            Output_mask = torch.ones([len(self.idx2relation.keys()), len(self.idx2entity.keys()), len(self.idx2entity.keys())], dtype=torch.float)
            if self.cuda:
                Output_mask = Output_mask.cuda()

            # gather objects, properties, and affordances
            task_names = ['situated-OP', 'situated-OA', 'situated-AP']
            task_mapping = defaultdict()

            for name in task_names:
                task_mapping[name] = Task(TASK_REV_MEDIUMHAND[name])

            objects, properties, affordances = get_entity_sets(task_mapping)

            for k in range(len(self.idx2relation.keys())):
                relation = self.idx2relation[k]
                for i in range(len(self.idx2entity.keys())):
                    for j in range(len(self.idx2entity.keys())):
                        e1 = self.idx2entity[i]
                        e2 = self.idx2entity[j]
                        if 'situated-OP' in relation:
                            if ((e1 in objects and e2 in properties) or (e2 in objects and e1 in properties)):
                                Output_mask[k, i, j] = 0.0
                        if 'situated-OA' in relation:
                            if ((e1 in objects and e2 in affordances) or (e2 in objects and e1 in affordances)):
                                Output_mask[k, i, j] = 0.0
                        if 'situated-AP' in relation:
                            if ((e1 in properties and e2 in affordances) or (e2 in properties and e1 in affordances)):
                                Output_mask[k, i, j] = 0.0

        train_data_idxs = self.get_data_idxs(d.train_data)
        print("Number of training data points: %d" % len(train_data_idxs))

        if self.add_dropout_bn==True:
            model = TuckER(d, self.ent_vec_dim, self.rel_vec_dim, **self.kwargs)
        else:
            model = TuckERNoDropoutBN(d, self.ent_vec_dim, self.rel_vec_dim, self.cuda)
        if self.cuda:
            model.cuda()
        model.init()
        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)

        er_vocab = self.get_er_vocab(train_data_idxs)
        er_vocab_pairs = list(er_vocab.keys())

        print("Starting training...")
        start_time = timeit.default_timer()
        for it in range(1, self.num_iterations+1):
            model.train()    
            losses = []
            np.random.shuffle(er_vocab_pairs)
            for j in range(0, len(er_vocab_pairs), self.batch_size):
                data_batch, targets = self.get_batch(er_vocab, er_vocab_pairs, j)
                opt.zero_grad()
                e1_idx = torch.tensor(data_batch[:,0])
                if self.do_link_prediction==True:
                    r_idx = torch.tensor(data_batch[:,1])
                else:
                    e2_idx = torch.tensor(data_batch[:,1])  
                if self.cuda:
                    e1_idx = e1_idx.cuda()
                    if self.do_link_prediction==True:
                        r_idx = r_idx.cuda()
                    else:
                        e2_idx = e2_idx.cuda()
                if self.do_link_prediction==True:
                    outputs = model.forward_lp(e1_idx, r_idx)
                else:
                    outputs = model.forward(e1_idx, e2_idx)

                if self.add_dropout_bn==True:
                    predictions = outputs
                else:
                    predictions = outputs[0]
                    W = outputs[1]
                    E = outputs[2]
                    R = outputs[3]
                if self.label_smoothing:
                    targets = ((1.0-self.label_smoothing)*targets) + (1.0/targets.size(1))

                loss = model.loss(predictions, targets)

                if self.add_constraint==True:
                    reg = self.reg

                    d1_want = R.size(0)
                    d2_want = E.size(0)
                    d3_want = d2_want
                    d1_in = R.size(1)
                    d2_in = E.size(1)
                    d3_in = d2_in

                    W_mat = torch.mm(E, W.view(d3_in, -1))
                    W_mat = W_mat.view(d1_in, d2_in, d3_want)

                    W_mat = torch.mm(E, W_mat.view(d2_in, -1))
                    W_mat = W_mat.view(d1_in, d2_want, d3_want)

                    W_mat = torch.mm(R, W_mat.view(d1_in, -1))
                    Output = W_mat.view(d1_want, d2_want, d3_want)

                    type_constraint = ((Output*Output_mask)**2).mean()
                    loss += (reg*type_constraint)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            if self.decay_rate:
                scheduler.step()

            print(it)
        stop_time = timeit.default_timer()
        print('training time: {}'.format((stop_time-start_time)/self.num_iterations))

        if self.do_link_prediction==True:
            model.eval()
            with torch.no_grad():
                print("Test:")
                self.evaluate_link_prediction(model, d.test_data)
        
        if not os.path.exists(self.saved_model_path):
            os.makedirs(self.saved_model_path)
        torch.save(model, os.path.join(self.saved_model_path, 'model.pt'))
        joblib.dump([self.entity_idxs, self.relation_idxs], os.path.join(self.saved_model_path, 'dic.pkl'))
           

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_iterations", type=int, default=20, nargs="?",
                    help="Number of iterations.")
    parser.add_argument("--batch_size", type=int, default=512, nargs="?",
                    help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.0005, nargs="?",
                    help="Learning rate.")
    parser.add_argument("--dr", type=float, default=1.0, nargs="?",
                    help="Decay rate.")
    parser.add_argument("--edim", type=int, default=200, nargs="?",
                    help="Entity embedding dimensionality.")
    parser.add_argument("--rdim", type=int, default=200, nargs="?",
                    help="Relation embedding dimensionality.")
    parser.add_argument("--cuda", type=bool, default=True if torch.cuda.is_available() else False, nargs="?",
                    help="Whether to use cuda (GPU) or not (CPU).")
    parser.add_argument("--input_dropout", type=float, default=0.3, nargs="?",
                    help="Input layer dropout.")
    parser.add_argument("--hidden_dropout1", type=float, default=0.4, nargs="?",
                    help="Dropout after the first hidden layer.")
    parser.add_argument("--hidden_dropout2", type=float, default=0.5, nargs="?",
                    help="Dropout after the second hidden layer.")
    parser.add_argument("--label_smoothing", type=float, default=0.1, nargs="?",
                    help="Amount of label smoothing.")
    parser.add_argument("--reg", type=float, default=0.1,)

    parser.add_argument("--data_dir", type=str, default='data/kge/generated/')
    parser.add_argument("--saved_model_path", type=str, default='pc_tucker_models')
    parser.add_argument("--add_constraint", action='store_true')
    parser.add_argument("--add_dropout_bn", action='store_true')
    parser.add_argument("--do_link_prediction", action='store_true')

    args = parser.parse_args()
    data_dir = args.data_dir
    torch.backends.cudnn.deterministic = True 
    seed = 20
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed) 
    d = Data(data_dir=data_dir, reverse=True, add_constraint=args.add_constraint)
    experiment = Experiment(num_iterations=args.num_iterations, batch_size=args.batch_size, learning_rate=args.lr, 
                            decay_rate=args.dr, ent_vec_dim=args.edim, rel_vec_dim=args.rdim, cuda=args.cuda,
                            input_dropout=args.input_dropout, hidden_dropout1=args.hidden_dropout1, hidden_dropout2=args.hidden_dropout2,
                            label_smoothing=args.label_smoothing, 
                            saved_model_path=args.saved_model_path,
                            add_constraint=args.add_constraint,
                            add_dropout_bn=args.add_dropout_bn,
                            do_link_prediction=args.do_link_prediction,
                            reg=args.reg)
    experiment.train()
                

