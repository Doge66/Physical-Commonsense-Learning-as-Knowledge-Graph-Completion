import openke
from openke.config import Trainer, Tester
from openke.module.model import RESCAL
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader

import joblib
import torch
import numpy as np
from collections import defaultdict
import argparse
import os
import sys
import timeit

from data import (
    TASK_REV_MEDIUMHAND,
    TASK_LABELS,
)
import metrics

if not os.path.exists('checkpoint'):
    os.makedirs('checkpoint')
    
# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./data/kge/openke/", 
	nbatches = 100,
	threads = 8, 
	sampling_mode = "normal", 
	bern_flag = 1, 
	filter_flag = 1, 
	neg_ent = 25,
	neg_rel = 0
)

# dataloader for test
test_dataloader = TestDataLoader("./data/kge/openke/", "link")

# define the model
rescal = RESCAL(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 50
)

# define the loss function
model = NegativeSampling(
	model = rescal, 
	loss = MarginLoss(margin = 1.0),
	batch_size = train_dataloader.get_batch_size(), 
)

start_time = timeit.default_timer()

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 0.1, use_gpu = True, opt_method = "adagrad")
trainer.run()
rescal.save_checkpoint('./checkpoint/rescal.ckpt')

stop_time = timeit.default_timer()
print('average training time: {}'.format((stop_time-start_time)/1000))

start_time = timeit.default_timer()

# test the model
rescal.load_checkpoint('./checkpoint/rescal.ckpt')
tester = Tester(model = rescal, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)

stop_time = timeit.default_timer()
print('link prediction testing time: {}'.format((stop_time-start_time)))