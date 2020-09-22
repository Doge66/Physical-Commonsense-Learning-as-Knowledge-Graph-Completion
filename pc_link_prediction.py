import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default='./data/kge/generated/')
parser.add_argument("--add_constraint", action='store_true')
args = parser.parse_args()

if args.add_constraint==True:
    os.system('python tucker_train.py --data_dir {:s} --saved_model_path {:s} --num_iterations {:d} --add_constraint --do_link_prediction'.format(args.data_dir, 'pc_lp_models', 50))
else:
    os.system('python tucker_train.py --data_dir {:s} --saved_model_path {:s} --num_iterations {:d} --do_link_prediction'.format(args.data_dir, 'pc_lp_models', 50))