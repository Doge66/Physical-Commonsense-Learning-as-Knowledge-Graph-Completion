import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default='./data/kge/original/')
args = parser.parse_args()

os.system('python tucker_train.py --data_dir {:s} --saved_model_path {:s} --num_iterations {:d} --add_dropout_bn --do_link_prediction'.format(args.data_dir, 'tucker_lp_models', 2000))
