import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import glob
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import numpy as np

import tqdm
import random
from modules.fsl_semi_query import make_semi_fsl
from dataloader import make_semi_dataloader
from utils import mean_confidence_interval, AverageMeter, set_seed


class semi_trainer(object):
    def __init__(self, cfg, checkpoint_dir):

        self.n_way = cfg.n_way  # 5
        self.k_shot = cfg.k_shot  # 5
        self.train_query_per_class = cfg.train.query_per_class_per_episode  # 10
        self.val_query_per_class = cfg.test.query_per_class_per_episode  # 15
        self.train_episode_per_epoch = cfg.train.episode_per_epoch
        self.prefix = osp.basename(checkpoint_dir)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = checkpoint_dir
        self.epochs = cfg.train.epochs

        self.fsl = make_semi_fsl(cfg).to(self.device)

        self.lr = cfg.train.learning_rate
        self.lr_decay = cfg.train.lr_decay
        self.lr_decay_epoch = cfg.train.lr_decay_epoch
        if cfg.train.optim == "Adam":
            self.optim = Adam(self.fsl.parameters(), lr=cfg.train.learning_rate, betas=cfg.train.adam_betas)
        elif cfg.train.optim == "SGD":
            self.optim = SGD(
                self.fsl.parameters(),
                lr=cfg.train.learning_rate,
                momentum=cfg.train.sgd_mom,
                weight_decay=cfg.train.sgd_weight_decay,
                nesterov=True
            )
        else:
            raise NotImplementedError
        self.val_episode = cfg.val.episode

        self.val_episode = cfg.val.episode
        pths = [osp.basename(f) for f in glob.glob(osp.join(checkpoint_dir, "*.pth"))]
        if pths:
            pths_epoch = [''.join(filter(str.isdigit, f[:f.find('_')])) for f in pths]
            pths = [p for p, e in zip(pths, pths_epoch) if e]
            pths_epoch = [int(e) for e in pths_epoch if e]
            self.train_start_epoch = max(pths_epoch)
            c = osp.join(checkpoint_dir, pths[pths_epoch.index(self.train_start_epoch)])
            state_dict = torch.load(c)
            self.fsl.load_state_dict(state_dict["fsl"])
            print("[*] Continue training from checkpoints: {}".format(c))
            lr_scheduler_last_epoch = self.train_start_epoch
        else:
            self.train_start_epoch = 0
            lr_scheduler_last_epoch = -1

        if cfg.train.lr_decay_milestones:
            self.lr_scheduler = MultiStepLR(self.optim, milestones=cfg.train.lr_decay_milestones, gamma=self.lr_decay)
        else:
            self.lr_scheduler = StepLR(self.optim, step_size=self.lr_decay_epoch, gamma=self.lr_decay)

        self.cfg = cfg

    def validate(self, dataloader):
        accuracies = []
        tqdm_gen = tqdm.tqdm(dataloader)
        acc = AverageMeter()
        for episode, (support_x, support_y, query_x, query_y, unlabeled_x) in enumerate(tqdm_gen):
            support_x = support_x.to(self.device)
            support_y = support_y.to(self.device)
            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device)
            unlabeled_x = unlabeled_x.to(self.device)

            rewards = self.fsl(support_x, support_y, query_x, query_y, unlabeled_x)
            total_rewards = np.sum(rewards)

            accuracy = total_rewards / (query_y.numel())

            acc.update(total_rewards / query_y.numel(), 1)
            mesg = "Val: acc={:.3f}".format(
                acc.avg
            )
            tqdm_gen.set_description(mesg)

            accuracies.append(accuracy)

        test_accuracy, h = mean_confidence_interval(accuracies)
        return test_accuracy, h

    def save_model(self, prefix, accuracy, h, epoch, final_epoch=False):
        filename = osp.join(self.checkpoint_dir, "e{}_{}way_{}shot.pth".format(prefix, self.n_way, self.k_shot))
        recordname = osp.join(self.checkpoint_dir, "e{}_{}way_{}shot.txt".format(prefix, self.n_way, self.k_shot))
        state = {
            'episode': prefix,
            'fsl': self.fsl.state_dict(),
            'epoch': epoch,
            "optimizer": None if not final_epoch else self.optim.state_dict()
        }
        with open(recordname, 'w') as f:
            f.write("prefix: {}\nepoch: {}\naccuracy: {}\nh: {}\n".format(prefix, epoch, accuracy, h))
        torch.save(state, filename)

    def train(self, dataloader, epoch):
        losses = AverageMeter()
        tqdm_gen = tqdm.tqdm(dataloader)
        for episode, (support_x, support_y, query_x, query_y, unlabeled_x) in enumerate(tqdm_gen):
            support_x = support_x.to(self.device)
            support_y = support_y.to(self.device)
            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device)
            unlabeled_x = unlabeled_x.to(self.device)

            loss = self.fsl(support_x, support_y, query_x, query_y, unlabeled_x)
            loss_sum = sum(loss.values())
            self.optim.zero_grad()
            loss_sum.backward()
            self.optim.step()
            losses.update(loss_sum.item(), len(query_x))

            mesg = "epoch {}, loss={:.3f}".format(
                epoch,
                losses.avg
            )
            tqdm_gen.set_description(mesg)

    def run(self):
        best_accuracy = 0.0
        # set_seed(1) # We run ten trials and should set same seed
        train_dataset = None
        val_dataloader, _ = make_semi_dataloader(self.cfg, phase="val", batch_size=self.cfg.test.batch_size,
                                                 last_dataset=None)
        for epoch in range(self.train_start_epoch, self.epochs):
            train_dataloader, train_dataset = make_semi_dataloader(
                self.cfg, phase="train",
                batch_size=self.cfg.train.batch_size,
                last_dataset=train_dataset
            )
            self.train(train_dataloader, epoch + 1)
            self.fsl.eval()
            with torch.no_grad():
                val_accuracy, h = self.validate(val_dataloader)

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                self.save_model("best", val_accuracy, h, epoch + 1, True)

            mesg = "\t Testing epoch {} validation accuracy: {:.3f}, h: {:.3f}".format(epoch + 1, val_accuracy, h)
            print(mesg)
            self.lr_scheduler.step()
            # self.save_model(epoch + 1, val_accuracy, h, epoch + 1, epoch == (self.epochs - 1))
            self.fsl.train()

