import os
import random

from cv_utils import eval_tc

data_path = 'data/kge/generated/'
n_fold = 5
reg = 10

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

t1_acc = 0
t1_micro_f1 = 0
t1_macro_f11 = 0
t1_macro_f12 = 0

t2_acc = 0
t2_micro_f1 = 0
t2_macro_f11 = 0
t2_macro_f12 = 0

t3_acc = 0
t3_micro_f1 = 0
t3_macro_f11 = 0
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

    os.system('python tucker_train.py --reg {} --data_dir {:s} --saved_model_path {:s} --add_constraint'.format(reg,
        os.path.join(data_path, 'cv/'), 'cv_pc_tc_models'))

    res = eval_tc('cv_pc_tc_models')

    t1_res = res[0]
    t2_res = res[1]
    t3_res = res[2]

    t1_acc += t1_res[0]
    t1_micro_f1 += t1_res[1]
    t1_macro_f11 += t1_res[2]
    t1_macro_f12 += t1_res[3]

    t2_acc += t2_res[0]
    t2_micro_f1 += t2_res[1]
    t2_macro_f11 += t2_res[2]
    t2_macro_f12 += t2_res[3]

    t3_acc += t3_res[0]
    t3_micro_f1 += t3_res[1]
    t3_macro_f11 += t3_res[2]
    t3_macro_f12 += t3_res[3]

    data = _shift(data, len_test)

print('-'*30)
print('task 1: acc: {:.02f}, micro f1: {:.02f}, macro f1 1: {:.02f}, macro f1 2: {:.02f}'.format(
    t1_acc/n_fold, t1_micro_f1/n_fold, t1_macro_f11/n_fold, t1_macro_f12/n_fold))
print('task 2: acc: {:.02f}, micro f1: {:.02f}, macro f1 1: {:.02f}, macro f1 2: {:.02f}'.format(
    t2_acc/n_fold, t2_micro_f1/n_fold, t2_macro_f11/n_fold, t2_macro_f12/n_fold))
print('task 3: acc: {:.02f}, micro f1: {:.02f}, macro f1 1: {:.02f}, macro f1 2: {:.02f}'.format(
    t3_acc/n_fold, t3_micro_f1/n_fold, t3_macro_f11/n_fold, t3_macro_f12/n_fold))