import logging
import random
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
'''
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)
'''
from warnings import filterwarnings
filterwarnings('ignore')
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import ttest_ind
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture
from torch import nn, optim
from utils.augment import Cutout, Solarize, Invert
from methods.er_baseline import ER
from utils.data_loader import cutmix_data, ImageDataset, StreamDataset, MemoryDataset, MyImageDataset
from utils.train_utils import select_model, soft_cross_entropy_loss
from utils.loss import coteaching_loss, dividemix_loss, neg_entropy_loss

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")

class ML(ER):
    def __init__(self, criterion, device, train_transform, test_transform, n_classes, additional_trans,  **kwargs):
        super().__init__(criterion, device, train_transform, test_transform, n_classes, **kwargs)
        self.memory_size = kwargs["memory_size"]

        # 추가사항
        self.robust_type = kwargs["robust_type"]
        self.warmup = kwargs["warm_up"]
        self.weak_transform = additional_trans

        # Samplewise importance variables
        self.loss = np.array([])
        self.dropped_idx = []
        self.memory_dropped_idx = []
        self.imp_update_counter = 0

        if self.robust_type =="PuriDivER":
            self.memory = MemoryDataset(self.dataset, self.train_transform, self.exposed_classes,
                                    test_transform=self.test_transform, data_dir=self.data_dir, device=self.device,
                                    transform_on_gpu=self.gpu_transform, robust_transform = [self.weak_transform, self.train_transform, self.test_transform], robust_test_transform = self.test_transform, save_test='cpu', keep_history=True)

        elif self.robust_type =="DivideMix":
            self.memory = MemoryDataset(self.dataset, self.train_transform, self.exposed_classes,
                                    test_transform=self.test_transform, data_dir=self.data_dir, device=self.device,
                                    transform_on_gpu=self.gpu_transform, robust_transform = [self.weak_transform, self.weak_transform, self.test_transform], robust_test_transform = self.test_transform, save_test='cpu', keep_history=True)
        else:
            self.memory = MemoryDataset(self.dataset, self.train_transform, self.exposed_classes,
                                    test_transform=self.test_transform, data_dir=self.data_dir, device=self.device,
                                    transform_on_gpu=self.gpu_transform, save_test='cpu', keep_history=True)

        self.imp_update_period = kwargs['imp_update_period']
        if kwargs["sched_name"] == 'default':
            self.sched_name = 'adaptive_lr'

        # Adaptive LR variables
        self.lr_step = kwargs["lr_step"]
        self.lr_length = kwargs["lr_length"]
        self.lr_period = kwargs["lr_period"]
        self.dataset_path = kwargs["dataset_path"]
        self.prev_loss = None
        self.lr_is_high = True
        self.high_lr = self.lr
        self.low_lr = self.lr_step * self.lr
        self.high_lr_loss = []
        self.low_lr_loss = []
        self.current_lr = self.lr


    def online_step(self, sample, sample_num, n_worker):
        #print("sample_num", sample_num)
        if sample['label'] not in self.exposed_classes:
            self.add_new_class(sample['label'])
        self.update_memory(sample)
        self.num_updates += self.online_iter
        #if self.num_updates >= self.batch_size:
        if self.num_updates >= 1:
            train_loss, train_acc = self.online_train([], self.batch_size, n_worker,
                                                      iterations=int(self.num_updates), stream_batch_size=0)
            self.report_training(sample_num, train_loss, train_acc)
            self.num_updates -= int(self.num_updates)
            self.update_schedule()

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
            df, self.dataset, data_dir=self.data_dir, transform=self.test_transform
        )
        dataloader = DataLoader(dataset, batch_size=min(test_batchsize, len(dataset)), shuffle=False)

        criterion = nn.CrossEntropyLoss(reduction='none')
        criterion = criterion.to(self.device)
        self.model.eval()

        with torch.no_grad():
            logits = []
            features = []
            labels = []
            #for batch_idx, data in enumerate(dataloader):
            batch_size = 32
            #print("batch_size", batch_size)
            for i in range(len(df)//batch_size+1):
                data = df.iloc[batch_size*i:min(batch_size*(i+1), len(df))]
                x = list(data["image"].values)
                y = list(data["label"].values)
                if len(x)==0:
                    break
                #print("x shape", x.shape)
                x = [i.tolist() for i in x]
                x = torch.Tensor(x)
                y = torch.Tensor(y).to(torch.long)
                x = x.to(self.device)
                y = y.to(self.device)
                #print("x!", x.shape)
                #print("y!", y.shape)
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


    def update_memory(self, sample):
        #print("mem imgs", len(self.memory.images), "memory_size", self.memory_size)
        if len(self.memory.datalist) >= self.memory_size:
            label_frequency = copy.deepcopy(self.memory.cls_count)
            label_frequency[self.exposed_classes.index(sample['label'])] += 1
            cls_to_replace = np.argmax(np.array(label_frequency))
            cand_idx = self.memory.cls_idx[cls_to_replace] # 여기에 max count label을 갖는 data들의 index가 들어있다.
            print("cand_idx_len", len(cand_idx))
            mem_images = list(np.array(self.memory.images)[cand_idx])
            mem_labels = list(np.array(self.memory.labels)[cand_idx])
            dic = {'image':mem_images, 'label':mem_labels, 'original_index':cand_idx}
            cand_df = pd.DataFrame(dic)
            cls_cand_df = self.calculate_loss_and_feature(cand_df)
            cls_loss = cls_cand_df["loss"].to_numpy()
            cls_loss = (cls_loss - cls_loss.mean()) / cls_loss.std()
            max_idx = np.argmax(cls_loss)
            min_idx = np.argmin(cls_loss)
            print("max", cls_loss[max_idx], "min", cls_loss[min_idx])
            if abs(cls_loss[max_idx])>abs(cls_loss[min_idx]):
                drop_idx = cand_df.iloc[max_idx]['original_index']
            else:
                drop_idx = cand_df.iloc[min_idx]['original_index']
            self.memory.replace_sample(sample, drop_idx)
        else:
            self.memory.replace_sample(sample)
        df = pd.DataFrame(self.memory.datalist)
        print(df.label.value_counts())

        if "true_label" in df.columns:
            n_clean = len(df[df["true_label"] == df["label"]])
        else:
            n_clean = len(df)

        print("Memory clean ratio", n_clean / len(df))


    def split_data(self, dataset, test_dataset, n, model=None):
        assert n in [2, 3], "N should be 2 or 3"
        if model is None:
            model = self.model

        CE = nn.CrossEntropyLoss(reduction='none')
        model.eval()
        test_dataset.reset_prob()
        loader = DataLoader(test_dataset,
                            shuffle=False,
                            batch_size=64,
                            num_workers=2,
                            )
        losses = torch.tensor([])
        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                inputs = data["image"]
                targets = data["label"]
                inputs, targets = inputs.cuda(), targets.cuda()
                outputs = model(inputs)
                loss = CE(outputs, targets)
                losses = torch.cat([losses, loss.detach().cpu()])
        
        if (losses.max() - losses.min())==0:
            losses = (losses - losses.min())
        else:
            losses = (losses - losses.min()) / (losses.max() - losses.min())

        input_loss = losses.reshape(-1, 1)

        # fit a two-component GMM to the loss
        gmm = GaussianMixture(n_components=n, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss)
        mean_index = np.argsort(gmm.means_, axis=0)
        prob = prob[:, mean_index]
        dataset.set_prob(prob.squeeze(axis=-1))
        # update!!
        #dataset = self.memory
        pred = prob.argmax(axis=1)

        idx = np.where(pred == 0)[0]
        correct_size = len(idx)
        if correct_size == 0:
            return [None for _ in range(n)]
        print("clean_check")
        print(dataset)
        print(pd.DataFrame(dataset.datalist).iloc[idx])
        dataloader_correct = DataLoader(torch.utils.data.Subset(dataset, idx),
                                        shuffle=True,
                                        batch_size=self.batch_size,
                                        num_workers=2
                                        )
        idx = np.where(pred == 1)[0]
        amb_size = len(idx)
        batch_size = int(amb_size / correct_size * self.batch_size)
        if batch_size < 2:
            batch_size = 2

        if amb_size <= 2:
            dataloader_ambiguous = None
        else:
            print("ambiguous_check")
            print(torch.utils.data.Subset(dataset, idx))
            dataloader_ambiguous = DataLoader(torch.utils.data.Subset(dataset, idx),
                                              shuffle=True,
                                              batch_size=batch_size,
                                              num_workers=2
                                              )

        if n == 3:
            idx = np.where(pred == 2)[0]
            incorrect_size = len(idx)
            batch_size = int(incorrect_size / correct_size * self.batch_size)
            if batch_size < 2:
                batch_size = 2
            if incorrect_size <= 2:
                dataloader_incorrect = None
            else:
                print("incorrect_check")
                print(torch.utils.data.Subset(dataset, idx))
                dataloader_incorrect = DataLoader(torch.utils.data.Subset(dataset, idx),
                                                  shuffle=True,
                                                  batch_size=batch_size,
                                                  num_workers=2
                                                  )

            logger.info(f"n_correct: {correct_size}\tn_ambiguous: {amb_size}\tn_incorrect: {incorrect_size}")
            return dataloader_correct, dataloader_ambiguous, dataloader_incorrect
        logger.info(f"n_correct: {correct_size}\tn_ambiguous: {amb_size}")
        return dataloader_correct, dataloader_ambiguous


    def puridiver_split(self, epoch, dataset, n, plot_gmm=False):
        assert n in [2], "N should be 2"
        dataloader_correct, dataloader_ambiguous, dataloader_incorrect = None, None, None
        CE = nn.CrossEntropyLoss(reduction='none')
        SM = torch.nn.Softmax(dim=1)
        self.model.eval()
        #print("len", len(dataset))
        dataset.reset_prob()
        loader = DataLoader(dataset,
                            shuffle=True,
                            batch_size=min(len(self.memory.datalist),64),
                            num_workers=2,
                            )
        losses = torch.tensor([])
        uncertainties = torch.tensor([])
        if plot_gmm:
            clean_noises = torch.tensor([], dtype=torch.bool)  # for plot
            cert_uncerts = torch.tensor([], dtype=torch.bool)  # for plot
        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                inputs = data["test_img"]
                targets = data["label"]
                #y_true = data["true_label"]  # for plotting.
                inputs = inputs.cuda()
                outputs = self.model(inputs)
                logits = SM(outputs)
                # uncertainty는 가장 확률이 높은 애와 1 사이의 차이
                # noise set을 2개로 나누기 위한 과정
                uncerts = 1 - torch.max(logits, 1)[0]

                targets = targets.cuda()
                loss = CE(outputs, targets)
                # list의 append와 같은 개념
                losses = torch.cat([losses, loss.detach().cpu()])
                uncertainties = torch.cat([uncertainties, uncerts.detach().cpu()])

        if (losses.max() - losses.min())==0:
            losses = (losses - losses.min())
        else:
            losses = (losses - losses.min()) / (losses.max() - losses.min())
        #print("after losses")
        #print(losses)
        input_loss = losses.reshape(-1, 1)
        uncertainties = uncertainties.reshape(-1, 1)

        # fit a two-component GMM to the loss
        # clean과 noisy로 구분하는 과정
        gmm_loss = GaussianMixture(n_components=n, max_iter=10, tol=1e-2, reg_covar=5e-4)
        gmm_loss.fit(input_loss)
        gmm_loss_means = gmm_loss.means_
        if gmm_loss_means[0] <= gmm_loss_means[1]:
            small_loss_idx = 0
            large_loss_idx = 1
        else:
            small_loss_idx = 1
            large_loss_idx = 0

        prob = gmm_loss.predict_proba(input_loss)
        dataset.set_prob(prob)
        # update!
        #dataset = self.memory
        pred = prob.argmax(axis=1)

        idx = np.where(pred == small_loss_idx)[0]
        correct_size = len(idx)
        if correct_size == 0:
            return None, None, None

        # clean 데이터
        #print("clean_check")
        #print(torch.utils.data.Subset(dataset, idx))
        dataloader_correct = DataLoader(torch.utils.data.Subset(dataset, idx),
                                        shuffle=True,
                                        batch_size=self.batch_size,
                                        num_workers=2,
                                        )
        # 2nd GMM using large loss datasets
        # 이제 noisy를 갖고 relabel/unlabel로 또 split하자
        idx = np.where(pred == large_loss_idx)[0]
        high_loss_size = len(idx)
        batch_size = int(high_loss_size / correct_size * self.batch_size)
        if batch_size < 2:
            batch_size = 2

        if high_loss_size <= 2:
            dataloader_ambiguous = None
            dataloader_incorrect = None
        else:
            # fit a two-component GMM to the loss
            # relabeling set / unlabeled set으로 split하기 위함
            gmm_uncert = GaussianMixture(n_components=n, max_iter=10, tol=1e-2, reg_covar=5e-4)
            gmm_uncert.fit(uncertainties[idx])
            prob_uncert = gmm_uncert.predict_proba(uncertainties[idx])
            pred_uncert = prob_uncert.argmax(axis=1)

            if gmm_uncert.means_[0] <= gmm_uncert.means_[1]:
                small_loss_idx = 0
                large_loss_idx = 1
            else:
                small_loss_idx = 1
                large_loss_idx = 0

            # relabeing을 진행할 set
            idx_uncert = np.where(pred_uncert == small_loss_idx)[0]
            amb_size = len(idx_uncert)
            batch_size = int(amb_size / correct_size * self.batch_size)
            if batch_size < 2:
                batch_size = 2

            if amb_size <= 2:
                dataloader_ambiguous = None
            else:
                #print("ambiguous_check")
                #print(torch.utils.data.Subset(dataset, idx[idx_uncert]))
                dataloader_ambiguous = DataLoader(torch.utils.data.Subset(dataset, idx[idx_uncert]),
                                                  shuffle=True,
                                                  batch_size=batch_size,
                                                  num_workers=2,
                                                  )
            # unlabeled training에 쓰일 set
            idx_uncert = np.where(pred_uncert == large_loss_idx)[0]
            incorrect_size = len(idx_uncert)
            batch_size = int(incorrect_size / correct_size * self.batch_size)
            if batch_size < 2:
                batch_size = 2
            if incorrect_size <= 2:
                dataloader_incorrect = None
            else:
                #print("unlabeled_check")
                #print(torch.utils.data.Subset(dataset, idx[idx_uncert]))
                dataloader_incorrect = DataLoader(torch.utils.data.Subset(dataset, idx[idx_uncert]),
                                                  shuffle=True,
                                                  batch_size=batch_size,
                                                  num_workers=2,
                                                  )
            logger.info(f"n_correct: {correct_size}\tn_ambiguous: {amb_size}\tn_incorrect: {incorrect_size}")
        logger.info(f"n_correct: {correct_size}\tn_high_loss: {high_loss_size}")

        return dataloader_correct, dataloader_ambiguous, dataloader_incorrect


    def dividemix(self, epoch, net, net2, optimizer, labeled_trainloader, unlabeled_trainloader, warm_up=10, T=0.5,
                  alpha=4):
        net.train()
        net2.eval()  # fix one network and train the other

        num_iter = 0 if labeled_trainloader is None else len(labeled_trainloader)
        use_unlabeled = unlabeled_trainloader is not None

        if num_iter > 0:
            if use_unlabeled:
                unlabeled_train_iter = iter(unlabeled_trainloader)
            for batch_idx, data in enumerate(labeled_trainloader):
                inputs_x = data['image']
                inputs_x2 = data['origin_img']
                labels_x = data['label']
                if len(labels_x) <= 1:
                    continue
                w_x = data['prob'][:, 0]
                if use_unlabeled:
                    try:
                        data = unlabeled_train_iter.next()
                        inputs_u = data['image']
                        inputs_u2 = data['origin_img']
                    except:
                        unlabeled_train_iter = iter(unlabeled_trainloader)
                        data = unlabeled_train_iter.next()
                        inputs_u = data['image']
                        inputs_u2 = data['origin_img']
                    if inputs_u.size(0) <= 1:
                        continue

                batch_size = inputs_x.size(0)

                # Transform label to one-hot
                labels_x = torch.zeros(batch_size, self.num_learning_class).scatter_(1, labels_x.view(-1, 1), 1)
                w_x = w_x.view(-1, 1).type(torch.FloatTensor)

                inputs_x, inputs_x2, labels_x, w_x = inputs_x.cuda(), inputs_x2.cuda(), labels_x.cuda(), w_x.cuda()
                if use_unlabeled:
                    inputs_u, inputs_u2 = inputs_u.cuda(), inputs_u2.cuda()
                with torch.no_grad():
                    # label co-guessing of unlabeled samples
                    if use_unlabeled:
                        outputs_u11 = net(inputs_u)
                        outputs_u12 = net(inputs_u2)
                        outputs_u21 = net2(inputs_u)
                        outputs_u22 = net2(inputs_u2)

                        pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(
                            outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4
                        ptu = pu ** (1 / T)  # temparature sharpening

                        targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
                        targets_u = targets_u.detach()

                    # label refinement of labeled samples
                    outputs_x = net(inputs_x)
                    outputs_x2 = net(inputs_x2)

                    px = (torch.softmax(outputs_x, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
                    px = w_x * labels_x + (1 - w_x) * px
                    ptx = px ** (1 / T)  # temparature sharpening

                    targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
                    targets_x = targets_x.detach()

                    # mixmatch
                l = np.random.beta(alpha, alpha)
                l = max(l, 1 - l)

                if use_unlabeled:
                    all_inputs = torch.cat([inputs_x, inputs_x2, inputs_u, inputs_u2], dim=0)
                    all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)
                else:
                    all_inputs = torch.cat([inputs_x, inputs_x2], dim=0)
                    all_targets = torch.cat([targets_x, targets_x], dim=0)

                idx = torch.randperm(all_inputs.size(0))

                input_a, input_b = all_inputs, all_inputs[idx]
                target_a, target_b = all_targets, all_targets[idx]

                mixed_input = l * input_a + (1 - l) * input_b
                mixed_target = l * target_a + (1 - l) * target_b

                logits = net(mixed_input)
                logits_x = logits[:batch_size * 2]
                if use_unlabeled:
                    logits_u = logits[batch_size * 2:]
                    Lx, Lu, lamb = dividemix_loss(logits_x, mixed_target[:batch_size * 2], logits_u,
                                                  mixed_target[batch_size * 2:],
                                                  epoch + batch_idx / num_iter, warm_up)

                else:
                    Lx, Lu, lamb = dividemix_loss(logits_x, mixed_target[:batch_size * 2], torch.Tensor([[0]]),
                                                  torch.Tensor([[0]]),
                                                  epoch + batch_idx / num_iter, warm_up)

                # regularization
                prior = torch.ones(self.num_learning_class) / self.num_learning_class
                prior = prior.cuda()
                pred_mean = torch.softmax(logits, dim=1).mean(0)
                penalty = torch.sum(prior * torch.log(prior / pred_mean))

                loss = Lx + lamb * Lu + penalty
                # compute gradient and do SGD step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


    def puridiver(self, loader_L, loader_U, loader_R, optimizer):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        criterion_L = torch.nn.CrossEntropyLoss()
        criterion_U = torch.nn.MSELoss()

        unlabeled_train_iter = iter(loader_U)
        relabeled_train_iter = iter(loader_R)

        self.model.train()
        for i, data in enumerate(loader_L):
            if i==1:
                break
            x_l = data["image"]
            y_l = data["label"]

            try:
                data_r = relabeled_train_iter.next()
            except:
                relabeled_train_iter = iter(loader_R)
                data_r = relabeled_train_iter.next()
            try:
                data_u = unlabeled_train_iter.next()
            except:
                unlabeled_train_iter = iter(loader_U)
                data_u = unlabeled_train_iter.next()
            x_pseudo = data_r["origin_img"]
            x_r = data_r["image"]
            y_r = data_r["label"]
            y_r = torch.nn.functional.one_hot(y_r, num_classes=self.num_learned_class)
            correct_prob = data_r["prob"][:, 0]
            correct_prob = correct_prob.unsqueeze(axis=1).expand(-1, self.num_learned_class)
            x_u_weak = data_u["origin_img"] # transform 전
            x_u_strong = data_u["image"] # transform 후

            y_l = y_l.to(self.device)

            optimizer.zero_grad()

            do_cutmix = self.cutmix and np.random.rand(1) < 0.5
            if do_cutmix:
                x_cutmix, labels_a, labels_b, lam = cutmix_data(x=x_l, y=y_l, alpha=1.0)
                x_concat = torch.cat([x_pseudo, x_r, x_u_weak, x_u_strong, x_cutmix])
                x_concat = x_concat.to(self.device)
                logit = self.model(x_concat)
                r_size = x_pseudo.size(0)
                u_size = x_u_weak.size(0)
                logit_pseudo, logit_r, logit_u_weak, logit_u_strong, logit_cutmix = \
                    logit[:r_size], logit[r_size:2 * r_size], logit[2 * r_size: 2 * r_size + u_size], \
                    logit[2 * r_size + u_size:2 * r_size + 2 * u_size], logit[2 * r_size + 2 * u_size:]
                logit_pseudo_softmax = torch.nn.functional.softmax(logit_pseudo, dim=1)

                loss_L = lam * criterion_L(logit_cutmix, labels_a) + (1 - lam) * criterion_L(
                    logit_cutmix, labels_b
                )
                
                soft_pseudo = correct_prob * y_r + (1 - correct_prob) * logit_pseudo_softmax.detach().cpu()
                soft_pseudo = soft_pseudo.to(self.device)

                loss_R = soft_cross_entropy_loss(logit_r, soft_pseudo)
                loss_U = criterion_U(logit_u_strong, logit_u_weak)
                loss = (y_l.size(0) / (y_l.size(0) + y_r.size(0) + u_size)) * loss_L + \
                       (y_r.size(0) / (y_l.size(0) + y_r.size(0) + u_size)) * loss_R + \
                       (u_size / (y_l.size(0) + y_r.size(0) + u_size)) * loss_U

                print(f"Loss L: {loss_L.item()} | Loss R: {loss_R.item()} | Loss U: {loss_U.item()}")
                _, preds = logit_cutmix.topk(self.topk, 1, True, True)
            else:
                x_concat = torch.cat([x_pseudo, x_r, x_u_weak, x_u_strong, x_l])
                x_concat = x_concat.to(self.device)
                logit = self.model(x_concat)
                r_size = x_pseudo.size(0)
                u_size = x_u_weak.size(0)
                logit_pseudo, logit_r, logit_u_weak, logit_u_strong, logit_l = \
                    logit[:r_size], logit[r_size:2 * r_size], logit[2 * r_size:2 * r_size + u_size], \
                    logit[2 * r_size + u_size:2 * r_size + 2 * u_size], logit[2 * r_size + 2 * u_size:]

                logit_pseudo_softmax = torch.nn.functional.softmax(logit_pseudo, dim=1)

                soft_pseudo = correct_prob * y_r + (1 - correct_prob) * logit_pseudo_softmax.detach().cpu()
                soft_pseudo = soft_pseudo.to(self.device)

                # loss는 각각 적용 (unlabel은 MSE loss, label과 relabel은 cross entropy 적용)
                loss = (y_l.size(0) / (y_l.size(0) + y_r.size(0) + u_size)) * criterion_L(logit_l, y_l) + \
                       (y_r.size(0) / (y_l.size(0) + y_r.size(0) + u_size)) * soft_cross_entropy_loss(logit_r,
                                                                                                      soft_pseudo) + \
                       (u_size / (y_l.size(0) + y_r.size(0) + u_size)) * criterion_U(logit_u_weak, logit_u_strong)

                _, preds = logit_l.topk(self.topk, 1, True, True)


            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            #loss.backward()
            #optimizer.step()
            total_loss += loss.item()
            correct += torch.sum(preds == y_l.unsqueeze(1)).item()
            num_data += y_l.size(0)

        #n_batches = len(loader_L)
        n_batches = 1
        return total_loss / n_batches, correct / num_data


    #ml에서는 사실 stream_batch_size가 0
    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=0):
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        if stream_batch_size > 0:
            sample_dataset = StreamDataset(sample, dataset=self.dataset, transform=self.train_transform,
                                           cls_list=self.exposed_classes, data_dir=self.data_dir, device=self.device,
                                           transform_on_gpu=True)
            #print("sample_dataset")
            #print(sample_dataset)
        if len(self.memory.datalist) > 0 and batch_size - stream_batch_size > 0:
            memory_batch_size = min(len(self.memory), batch_size - stream_batch_size)
        #print("memory_batch_size",memory_batch_size)
        for i in range(iterations):
            self.model.train()
            x = []
            y = []
            if stream_batch_size > 0:
                stream_data = sample_dataset.get_data()
                x.append(stream_data['image'])
                y.append(stream_data['label'])
            if len(self.memory.datalist) > 0 and batch_size - stream_batch_size > 0:
                '''
                if self.robust_type=="none":
                    memory_data = self.memory.get_batch(memory_batch_size)
                else:
                '''
                train_loss, train_acc = self.do_robust(i, n_worker, memory_batch_size)
        return train_loss, train_acc

    def do_robust(self, iteration, n_worker, memory_batch_size):
        mem_dataset = self.memory
        mem_dataset.reset_prob()
        memory_loader = DataLoader(
            mem_dataset,
            shuffle=True,
            batch_size=memory_batch_size,
            num_workers=n_worker,
        )
        epoch = 1
        #print("mem_dataset", len(mem_dataset))
        #print(mem_dataset)
        if self.robust_type == "DivideMix" and memory_batch_size>=2:
            print("div", iteration)
            label_loader, unlabel_loader = self.split_data(dataset=mem_dataset, test_dataset=split_dataset, n=2,
                                                           model=self.model_2)
            self.dividemix(epoch, self.model, self.model_2, self.optimizer, label_loader, unlabel_loader,
                           warm_up=self.warmup)
            label_loader, unlabel_loader = self.split_data(dataset=mem_dataset, test_dataset=split_dataset, n=2,
                                                           model=self.model)
            self.dividemix(epoch, self.model_2, self.model, self.optimizer_2, label_loader, unlabel_loader,
                           warm_up=self.warmup)

        elif self.robust_type == "PuriDivER" and memory_batch_size>=2:
            print("Puri", iteration)
            correct_loader, ambiguous_loader, incorrect_loader = self.puridiver_split(epoch, dataset=mem_dataset, n=2)
            if ambiguous_loader is not None and incorrect_loader is not None:
                train_loss, train_acc = self.puridiver(correct_loader, ambiguous_loader,
                                                       incorrect_loader,
                                                       optimizer=self.optimizer)
            else:
                train_loss, train_acc = self._train(memory_loader=memory_loader,
                                                    optimizer=self.optimizer, criterion=self.criterion)
        else:
            print("_train", iteration)
            train_loss, train_acc = self._train(memory_loader=memory_loader,
                                                optimizer=self.optimizer, criterion=self.criterion)
        return train_loss, train_acc 

    def _train(
            self, memory_loader, optimizer, criterion
    ):
        total_loss, correct, num_data = 0.0, 0.0, 0.0

        self.model.train()
        if self.robust_type == 'DivideMix':
            self.model_2.train()
        for idx, data in enumerate(memory_loader):
            if idx == 1:
                break
            x = data["image"].float()
            y = data["label"]
            x = x.to(self.device)
            y = y.to(self.device)

            l, c, d = self.update_model(x, y)
            total_loss += l
            correct += c
            num_data += d

        #n_batches = len(memory_loader)
        n_batches = 1
        return total_loss / n_batches, correct / num_data

    def update_model(self, x, y):
        self.optimizer.zero_grad()
        if self.robust_type == 'DivideMix':
            self.optimizer_2.zero_grad()

        do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            logit = self.model(x)
            loss = lam * self.criterion(logit, labels_a) + (1 - lam) * self.criterion(
                logit, labels_b
            )
            if self.robust_type in ['CoTeaching', 'DivideMix']:
                logit_2 = self.model_2(x)
                loss += lam * self.criterion(logit_2, labels_a) + (1 - lam) * self.criterion(
                    logit_2, labels_b
                )
        else:
            logit = self.model(x)
            #print("update_model_logit")
            #print(logit)
            loss = self.criterion(logit, y)
            if self.robust_type in ['CoTeaching', 'DivideMix']:
                logit_2 = self.model_2(x)
                loss += self.criterion(logit_2, y)
                if self.robust_type == 'DivideMix' and 'asymN' in self.exp_name:
                    loss += neg_entropy_loss(logit) + neg_entropy_loss(logit_2)

        _, preds = logit.topk(self.topk, 1, True, True)

        if self.use_amp:
            #print("update1!", loss)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            #print("update2!", loss)
            loss.backward()
            self.optimizer.step()

        if self.robust_type == 'DivideMix':
            self.optimizer_2.step()
        #print("preds")
        #print(preds)
        #print("unsqueeze")
        #print(y.unsqueeze(1))
        return loss.item(), torch.sum(preds == y.unsqueeze(1)).item(), y.size(0)


    def add_new_class(self, class_name):
        self.exposed_classes.append(class_name)
        #print("exposed", self.exposed_classes)
        self.num_learned_class = len(self.exposed_classes)
        prev_weight = copy.deepcopy(self.model.fc.weight.data)

        # define new fc layer
        #print("num_learned_class", self.num_learned_class)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_learned_class).to(self.device)

        # store previous fc layer's weight
        with torch.no_grad():
            if self.num_learned_class > 1:
                self.model.fc.weight[:self.num_learned_class - 1] = prev_weight
        
        # optimizer state copy
        #sdict = copy.deepcopy(self.optimizer.state_dict())

        # In param_group, storeing lr, weight decay etc..
        # fc_params = sdict['param_groups'][1]['params']


        del self.optimizer.param_groups[1]
        self.optimizer.add_param_group({'params': self.model.fc.parameters()})

        self.memory.add_new_class(cls_list=self.exposed_classes)
        if 'reset' in self.sched_name:
            self.update_schedule(reset=True)

    def update_schedule(self, reset=False):
        if self.sched_name == 'adaptive_lr':
            self.adaptive_lr(period=self.lr_period, min_iter=self.lr_length)
            self.model.train()
        else:
            super().update_schedule(reset)

    def samplewise_loss_update(self, ema_ratio=0.90, batchsize=512):
        self.imp_update_counter += 1
        if self.imp_update_counter % self.imp_update_period == 0:
            if len(self.memory) > 0:
                self.model.eval()
                with torch.no_grad():
                    x = self.memory.device_img
                    y = torch.LongTensor(self.memory.labels)
                    y = y.to(self.device)
                    if self.use_amp:
                        with torch.cuda.amp.autocast():
                            logit = torch.cat(
                                [self.model(torch.cat(x[i * batchsize:min((i + 1) * batchsize, len(x))]).to(self.device))
                                for i in range(-(-len(x) // batchsize))], dim=0)

                    else:
                        logit = torch.cat(
                            [self.model(torch.cat(x[i * batchsize:min((i + 1) * batchsize, len(x))]).to(self.device))
                             for i in range(-(-len(x) // batchsize))], dim=0)

                    loss = F.cross_entropy(logit, y, reduction='none').cpu().numpy()
                self.memory.update_loss_history(loss, self.loss, ema_ratio=ema_ratio, dropped_idx=self.memory_dropped_idx)
                self.memory_dropped_idx = []
                self.loss = loss

    def samplewise_importance_memory(self, sample):
        if len(self.memory.images) >= self.memory_size:
            label_frequency = copy.deepcopy(self.memory.cls_count)
            label_frequency[self.exposed_classes.index(sample['label'])] += 1
            cls_to_replace = np.argmax(np.array(label_frequency))
            cand_idx = self.memory.cls_idx[cls_to_replace]
            score = self.memory.others_loss_decrease[cand_idx]
            idx_to_replace = cand_idx[np.argmin(score)]
            self.memory.replace_sample(sample, idx_to_replace)
            self.dropped_idx.append(idx_to_replace)
            self.memory_dropped_idx.append(idx_to_replace)
        else:
            self.memory.replace_sample(sample)
            self.dropped_idx.append(len(self.memory) - 1)
            self.memory_dropped_idx.append(len(self.memory) - 1)

    def adaptive_lr(self, period=10, min_iter=10, significance=0.05):
        if self.imp_update_counter % self.imp_update_period == 0:
            self.train_count += 1
            mask = np.ones(len(self.loss), bool)
            mask[self.dropped_idx] = False
            if self.train_count % period == 0:
                if self.lr_is_high:
                    if self.prev_loss is not None and self.train_count > 20:
                        self.high_lr_loss.append(np.mean((self.prev_loss - self.loss[:len(self.prev_loss)])[mask[:len(self.prev_loss)]]))
                        if len(self.high_lr_loss) > min_iter:
                            del self.high_lr_loss[0]
                    self.prev_loss = self.loss
                    self.lr_is_high = False
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.low_lr
                        param_group["initial_lr"] = self.low_lr
                else:
                    if self.prev_loss is not None and self.train_count > 20:
                        self.low_lr_loss.append(np.mean((self.prev_loss - self.loss[:len(self.prev_loss)])[mask[:len(self.prev_loss)]]))
                        if len(self.low_lr_loss) > min_iter:
                            del self.low_lr_loss[0]
                    self.prev_loss = self.loss
                    self.lr_is_high = True
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.high_lr
                        param_group["initial_lr"] = self.high_lr
                self.dropped_idx = []
                if len(self.high_lr_loss) == len(self.low_lr_loss) and len(self.high_lr_loss) >= min_iter:
                    stat, pvalue = ttest_ind(self.low_lr_loss, self.high_lr_loss, equal_var=False, alternative='greater')
                    print(pvalue)
                    if pvalue < significance:
                        self.high_lr = self.low_lr
                        self.low_lr *= self.lr_step
                        self.high_lr_loss = []
                        self.low_lr_loss = []
                        if self.lr_is_high:
                            self.lr_is_high = False
                            for param_group in self.optimizer.param_groups:
                                param_group["lr"] = self.low_lr
                                param_group["initial_lr"] = self.low_lr
                        else:
                            self.lr_is_high = True
                            for param_group in self.optimizer.param_groups:
                                param_group["lr"] = self.high_lr
                                param_group["initial_lr"] = self.high_lr
                    elif pvalue > 1 - significance:
                        self.low_lr = self.high_lr
                        self.high_lr /= self.lr_step
                        self.high_lr_loss = []
                        self.low_lr_loss = []
                        if self.lr_is_high:
                            self.lr_is_high = False
                            for param_group in self.optimizer.param_groups:
                                param_group["lr"] = self.low_lr
                                param_group["initial_lr"] = self.low_lr
                        else:
                            self.lr_is_high = True
                            for param_group in self.optimizer.param_groups:
                                param_group["lr"] = self.high_lr
                                param_group["initial_lr"] = self.high_lr
