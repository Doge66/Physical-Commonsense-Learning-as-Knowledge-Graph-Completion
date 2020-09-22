import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default='./data/kge/original/')
args = parser.parse_args()

os.system('python tucker_train.py --data_dir {:s} --saved_model_path {:s} --add_dropout_bn'.format(args.data_dir, 'tucker_tc_models'))

os.system('python tucker_eval.py --saved_model_path {:s} --add_dropout_bn'.format('tucker_tc_models'))