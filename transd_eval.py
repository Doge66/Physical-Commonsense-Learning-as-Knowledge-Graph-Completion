import openke
from openke.config import Trainer, Tester
from openke.module.model import TransD
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
import time

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
	neg_rel = 0)

# dataloader for test
test_dataloader = TestDataLoader("./data/kge/openke/", "link")

# define the model
transd = TransD(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim_e = 200, 
	dim_r = 200, 
	p_norm = 1, 
	norm_flag = True)

# define the loss function
model = NegativeSampling(
	model = transd, 
	loss = MarginLoss(margin = 4.0),
	batch_size = train_dataloader.get_batch_size()
)

start_time = timeit.default_timer()

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1000, alpha = 1.0, use_gpu = True)
trainer.run()
transd.save_checkpoint('./checkpoint/transd.ckpt')

stop_time = timeit.default_timer()
print('average training time: {}'.format((stop_time-start_time)/1000))

start_time = timeit.default_timer()

# test the model
transd.load_checkpoint('./checkpoint/transd.ckpt')
tester = Tester(model = transd, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = False)

stop_time = timeit.default_timer()
print('link prediction testing time: {}'.format((stop_time-start_time)))