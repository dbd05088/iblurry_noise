import logging
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.augment import Cutout, Invert, Solarize, select_autoaugment
from torchvision import transforms
from randaugment.randaugment import RandAugment

from methods.er_baseline import ER
from utils.data_loader import cutmix_data, ImageDataset
from utils.augment import Cutout, Invert, Solarize, select_autoaugment

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i


class PuriDivER(ER):
    def __init__(
        self, criterion, device, train_transform, test_transform, n_classes, **kwargs
    ):
        super().__init__(
            criterion, device, train_transform, test_transform, n_classes, **kwargs
        )
        self.sched_name = "const"
        self.batch_size = kwargs["batchsize"]
        self.memory_epoch = kwargs["memory_epoch"]
        self.n_worker = kwargs["n_worker"]
        self.dataset_path = kwargs["dataset_path"]
        self.data_cnt = 0

    def online_step(self, sample, sample_num, n_worker):
        if sample['klass'] not in self.exposed_classes:
            self.add_new_class(sample['klass'])

        self.temp_batch.append(sample)
        self.update_memory(sample)
        self.num_updates += self.online_iter
        if self.num_updates >= 1:
            train_loss, train_acc = self.online_train([], self.batch_size, n_worker,
                                                      iterations=int(self.num_updates), stream_batch_size=0)
            self.report_training(sample_num, train_loss, train_acc)
            self.num_updates -= int(self.num_updates)
            self.update_schedule()

    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=0):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        if stream_batch_size > 0:
            sample_dataset = StreamDataset(sample, dataset=self.dataset, transform=self.train_transform,
                                           cls_list=self.exposed_classes, data_dir=self.data_dir, device=self.device,
                                           transform_on_gpu=True)
            print("sample_dataset")
            print(sample_dataset)
        if len(self.memory) > 0 and batch_size - stream_batch_size > 0:
            memory_batch_size = min(len(self.memory), batch_size - stream_batch_size)
        print("memory_batch_size",memory_batch_size)
        for i in range(iterations):
            self.model.train()
            x = []
            y = []
            if stream_batch_size > 0:
                stream_data = sample_dataset.get_data()
                x.append(stream_data['image'])
                y.append(stream_data['label'])
            if len(self.memory) > 0 and batch_size - stream_batch_size > 0:
                memory_data = self.memory.get_batch(memory_batch_size)
                x.append(memory_data['image'])
                y.append(memory_data['label'])
            x = torch.cat(x)
            y = torch.cat(y)

            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            logit, loss = self.model_forward(x, y)
            _, preds = logit.topk(self.topk, 1, True, True)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            self.samplewise_loss_update()

            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)

    def add_new_class(self, class_name):
        self.exposed_classes.append(class_name)
        self.num_learned_class = len(self.exposed_classes)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_learned_class).to(self.device)
        self.memory.add_new_class(cls_list=self.exposed_classes)
        self.reset_opt()

    def get_cls_cos_score(self, df):
        cls_features = np.array(df["feature"].tolist())

        weights = self.model.fc.block[0].weight.data.detach().cpu()
        clslist = df["label"].unique().tolist()
        assert len(clslist) == 1
        cls = clslist[0]
        relevant_idx = weights[cls, :] > torch.mean(weights, dim=0)

        cls_features = cls_features[:, relevant_idx]

        sim_matrix = cosine_similarity(cls_features)
        sim_score = sim_matrix.mean(axis=1)

        df['similarity'] = sim_score
        return df


    def calculate_loss_and_feature(self, df, get_loss=True, get_feature=True, test_batchsize=256):
        dataset = ImageDataset(
            df, self.dataset, data_dir=self.data_dir, cls_list=self.exposed_classes,transform=self.test_transform
        )
        dataloader = DataLoader(dataset, batch_size=min(test_batchsize, len(dataset)), shuffle=False)

        criterion = nn.CrossEntropyLoss(reduction='none')
        criterion = criterion.to(self.device)
        self.model.eval()

        with torch.no_grad():
            logits = []
            features = []
            labels = []
            for batch_idx, data in enumerate(dataloader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)
                logit, feature = self.model(x, get_feature=True)
                logits.append(logit)
                features.append(feature)
                labels.append(y)
            logits = torch.cat(logits, dim=0)
            features = torch.cat(features, dim=0)
            labels = torch.cat(labels, dim=0)

            if get_loss:
                loss = criterion(logits, labels)
                loss = loss.detach().cpu()
                loss = loss.tolist()
                df["loss"] = loss
            if get_feature:
                features = features.detach().cpu()
                features = features.tolist()
                df["feature"] = features
        return df

    # sample에 current data가 담겨 있음
    def update_memory(self, sample):
        #print("mem imgs", len(self.memory.images), "memory_size", self.memory_size)
        if len(self.memory.images) >= self.memory_size:
            print("drop!")
            cand_list =copy.deepcopy(self.memory.datalist)
            cand_list.append(sample)
            cand_df = pd.DataFrame(cand_list)
            cand_df = self.calculate_loss_and_feature(cand_df)
            cls_cnt = cand_df["label"].value_counts()
            cls_to_drop = cls_cnt[cls_cnt == cls_cnt.max()].sample().index[0] 
            cls_cand_df = cand_df[cand_df["label"] == cls_to_drop].copy()
            #cls_cand_df = self.get_cls_cos_score(cls_cand_df)
            cls_loss = cls_cand_df["loss"].to_numpy()
            cls_loss = (cls_loss - cls_loss.mean()) / cls_loss.std()
            max_idx = np.argmax(cls_loss)
            min_idx = np.argmin(cls_loss)
            #print("max", cls_loss[max_idx], "min", cls_loss[min_idx])
            if abs(cls_loss[max_idx])>abs(cls_loss[min_idx]):
                drop_idx = max_idx
            else:
                drop_idx = min_idx
            self.memory.replace_sample(sample, drop_idx)
        else:
            self.memory.replace_sample(sample)

    def online_before_task(self, cur_iter):
        self.reset_opt()
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda iter: 1)
'''
    def online_after_task(self, cur_iter):
        self.reset_opt()
        self.online_memory_train(
            cur_iter=cur_iter,
            n_epoch=self.memory_epoch,
            batch_size=self.batch_size,
        )
'''
    def online_memory_train(self, cur_iter, n_epoch, batch_size):
        if self.dataset == 'imagenet':
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, milestones=[30, 60, 80, 90], gamma=0.1
            )
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=1, T_mult=2, eta_min=self.lr * 0.01
            )
        mem_dataset = ImageDataset(
            pd.DataFrame(self.memory.datalist),
            dataset=self.dataset,
            transform=self.train_transform,
            cls_list=self.exposed_classes,
            data_dir=self.data_dir,
            preload=True,
            device=self.device,
            transform_on_gpu=True
        )
        for epoch in range(n_epoch):
            if epoch <= 0:  # Warm start of 1 epoch
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr * 0.1
            elif epoch == 1:  # Then set to maxlr
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.lr
            else:  # Aand go!
                self.scheduler.step()
            total_loss, correct, num_data = 0.0, 0.0, 0.0

            idxlist = mem_dataset.generate_idx(batch_size)
            for idx in idxlist:
                data = mem_dataset.get_data_gpu(idx)
                x = data["image"]
                y = data["label"]

                x = x.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()

                logit, loss = self.model_forward(x, y)
                _, preds = logit.topk(self.topk, 1, True, True)

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                    self.optimizer.step()
                total_loss += loss.item()
                correct += torch.sum(preds == y.unsqueeze(1)).item()
                num_data += y.size(0)
            n_batches = len(idxlist)
            train_loss, train_acc = total_loss / n_batches, correct / num_data
            logger.info(
                f"Task {cur_iter} | Epoch {epoch + 1}/{n_epoch} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
                f"lr {self.optimizer.param_groups[0]['lr']:.4f}"
            )


