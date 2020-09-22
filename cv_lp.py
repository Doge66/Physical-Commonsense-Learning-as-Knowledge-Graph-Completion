import os
import random

from cv_utils import eval_lp

data_path = 'data/kge/generated/'
n_fold = 5

random.seed(666)

def _shift(seq, n):
    return seq[n:]+seq[:n]

if not os.path.exists(os.path.join(data_path, 'cv')):
    os.makedirs(os.path.join(data_path, 'cv'))

data = []
with open(os.path.join(data_path, 'train.txt'), 'r') as f:
    lines = f.readlines()
    for line in lines:
        data.append(line)

random.shuffle(data)

mrr = 0
hit10 = 0
hit3 = 0
hit1 = 0

t3_macro_f12 = 0

len_test = int(len(data)/n_fold)
for i in range(n_fold):
    test_data = data[:len_test]
    train_data = data[len_test+1:]

    with open(os.path.join(data_path, 'cv', 'train.txt'), 'w') as f:
        for line in train_data:
            h,r,t = line.split()
            f.write('{}\t{}\t{}\n'.format(h,r,t))

    with open(os.path.join(data_path, 'cv', 'test.txt'), 'w') as f:
        for line in test_data:
            h,r,t = line.split()
            f.write('{}\t{}\t{}\n'.format(h,r,t))

    os.system('python tucker_train.py --data_dir {:s} --saved_model_path {:s} --num_iterations {:d} --add_constraint --do_link_prediction'.format(
        os.path.join(data_path, 'cv/'), 'cv_pc_lp_models', 50))

    res = eval_lp('cv_pc_lp_models', os.path.join(data_path, 'cv/'))

    mrr += res[0]
    hit10 += res[1]
    hit3 += res[2]
    hit1 += res[3]

    data = _shift(data, len_test)

print('-'*30)
print('MRR: {:.03f}, Hits @10: {:.03f}, Hits @3: {:.03f}, Hits @1: {:.03f}'.format(
    mrr/n_fold, hit10/n_fold, hit3/n_fold,hit1/n_fold))