import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default='./data/kge/generated/')
parser.add_argument("--add_constraint", action='store_true')
args = parser.parse_args()

# train
if args.add_constraint==True:
    os.system('python tucker_train.py --data_dir {:s} --saved_model_path {:s} --add_constraint'.format(args.data_dir, 'pc_tc_models'))
else:
    os.system('python tucker_train.py --data_dir {:s} --saved_model_path {:s}'.format(args.data_dir, 'pc_tc_models'))

# evaluation
if args.add_constraint==True:
    os.system('python tucker_eval.py --saved_model_path {:s} --add_constraint'.format('pc_tc_models'))
else:
    os.system('python tucker_eval.py --saved_model_path {:s}'.format('pc_tc_models'))

