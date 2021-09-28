import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

import random
from modules.fsl_semi_query import make_semi_fsl
from dataloader import make_semi_dataloader
from utils import mean_confidence_interval, AverageMeter, set_seed

class semi_evaluator(object):
    def __init__(self, cfg, checkpoint_dir):

        self.n_way                 = cfg.n_way # 5
        self.k_shot                = cfg.k_shot # 5
        self.test_query_per_class   = cfg.test.query_per_class_per_episode  # 15

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.eval_epoch = osp.basename(checkpoint_dir)
        self.prediction_folder = osp.join(
            "./predictions/", osp.basename(checkpoint_dir[:checkpoint_dir.rfind("/")])
        )
        if not osp.exists(self.prediction_folder):
            os.mkdir(self.prediction_folder)

        self.prediction_dir = osp.join(
            self.prediction_folder,
            osp.basename(checkpoint_dir).replace(".pth", ".txt")
        )

        self.f_txt = open(self.prediction_dir, 'w')
        self.checkpoint_dir = checkpoint_dir

        self.fsl = make_semi_fsl(cfg).to(self.device)

        state_dict = torch.load(checkpoint_dir)
        self.fsl.load_state_dict(state_dict["fsl"])
        self.fsl.eval()

        self.test_episode = cfg.test.episode
        self.total_testtimes = cfg.test.total_testtimes

        self.cfg = cfg

    def run(self):
        total_accuracies = 0.0
        total_h = 0.0
        print("evaluation epoch: ", self.eval_epoch, file=self.f_txt)
        last_dataset = None
        # set_seed(1)
        for epoch in range(self.total_testtimes):
            test_dataloader, last_dataset = make_semi_dataloader(self.cfg, phase="test", batch_size=self.cfg.test.batch_size, last_dataset=last_dataset)
            tqdm_gen = tqdm(test_dataloader)
            acc = AverageMeter()
            accuracies = []
            for episode, (support_x, support_y, query_x, query_y, unlabeled_x) in enumerate(tqdm_gen):
                support_x   = support_x.to(self.device)
                support_y   = support_y.to(self.device)
                query_x     = query_x.to(self.device)
                query_y     = query_y.to(self.device)
                unlabeled_x = unlabeled_x.to(self.device)

                rewards = self.fsl(support_x, support_y, query_x, query_y, unlabeled_x)
                total_rewards = np.sum(rewards)
                accuracy = total_rewards / (query_y.numel())
                acc.update(accuracy, 1)
                mesg = "Acc={:.3f}".format(acc.avg)
                tqdm_gen.set_description(mesg)
                accuracies.append(accuracy)

            test_accuracy, h = mean_confidence_interval(accuracies)
            print("test accuracy:",test_accuracy,"h:",h)
            print("test_accuracy:", test_accuracy, "h:", h, file=self.f_txt)
            total_accuracies += test_accuracy
            total_h += h
        print("aver_accuracy:", total_accuracies/self.total_testtimes, "h:", total_h/self.total_testtimes)
        print("aver_accuracy:", total_accuracies/self.total_testtimes, "h:", total_h/self.total_testtimes, file=self.f_txt)

