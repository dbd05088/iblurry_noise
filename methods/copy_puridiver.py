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
from utils.data_loader import cutmix_data, ImageDataset, MyImageDataset, StreamDataset
from utils.augment import Cutout, Invert, Solarize, select_autoaugment

logger = logging.getLogger()
writer = SummaryWriter("tensorboard")


def cycle(iterable):
    # iterate with shuffling
    while True:
        for i in iterable:
            yield i


class PuriDivER(ER):
    def __init__(self, criterion, device, train_transform, test_transform, n_classes, additional_trans,  **kwargs):
        super().__init__(
            criterion, device, train_transform, test_transform, n_classes, **kwargs
        )
        self.weak_transform = additional_trans
        self.sched_name = "const"
        self.batch_size = kwargs["batchsize"]
        self.memory_epoch = kwargs["memory_epoch"]
        self.n_worker = kwargs["n_worker"]
        self.robust_type = kwargs["robust_type"]
        self.warmup = kwargs["warm_up"]
        #self.dataset_path = kwargs["dataset_path"]
        self.data_cnt = 0

    def online_step(self, sample, sample_num, n_worker):
        if sample['label'] not in self.exposed_classes:
            self.add_new_class(sample['label'])

        self.temp_batch.append(sample)

        if len(self.temp_batch) == self.batch_size:
            train_loss, train_acc = self.online_train(self.temp_batch, self.batch_size, n_worker,
                                                      iterations=1, stream_batch_size=self.batch_size)
            self.report_training(sample_num, train_loss, train_acc)
            for stored_sample in self.temp_batch:
                self.update_memory(stored_sample)
            self.temp_batch = []

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
                            batch_size=16,
                            num_workers=2,
                            )
        losses = torch.tensor([])
        uncertainties = torch.tensor([])
        if plot_gmm:
            clean_noises = torch.tensor([], dtype=torch.bool)  # for plot
            cert_uncerts = torch.tensor([], dtype=torch.bool)  # for plot
        with torch.no_grad():
            for batch_idx, data in enumerate(loader):
                #print("batch_idx", batch_idx)
                inputs = data["test_img"]
                targets = data["label"]
                y_true = data["true_label"]  # for plotting.
                inputs = inputs.cuda()
                outputs = self.model(inputs)
                logits = SM(outputs)
                # uncertainty??? ?????? ????????? ?????? ?????? 1 ????????? ??????
                # noise set??? 2?????? ????????? ?????? ??????
                uncerts = 1 - torch.max(logits, 1)[0]
                '''
                if plot_gmm:
                    clean_noises = torch.cat([clean_noises, (targets == y_true)])  # for plot
                    cert_uncerts = torch.cat(
                        [cert_uncerts, (outputs.detach().cpu().argmax(axis=1) == y_true)])  # for plot
                    # true_targets = torch.cat([true_targets, y_true])  # for plot
                    # pred_targets = torch.cat([pred_targets, outputs.detach().cpu()])
                '''
                targets = targets.cuda()
                loss = CE(outputs, targets)
                # list??? append??? ?????? ??????
                losses = torch.cat([losses, loss.detach().cpu()])
                uncertainties = torch.cat([uncertainties, uncerts.detach().cpu()])

        losses = (losses - losses.min()) / (losses.max() - losses.min())
        input_loss = losses.reshape(-1, 1)
        uncertainties = uncertainties.reshape(-1, 1)


        # fit a two-component GMM to the loss
        # clean??? noisy??? ???????????? ??????
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

        # clean ?????????
        dataloader_correct = DataLoader(torch.utils.data.Subset(dataset, idx),
                                        shuffle=True,
                                        batch_size=self.batch_size,
                                        num_workers=2,
                                        )
        # 2nd GMM using large loss datasets
        # ?????? noisy??? ?????? relabel/unlabel??? ??? split??????
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
            # relabeling set / unlabeled set?????? split?????? ??????
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

            # relabeing??? ????????? set
            idx_uncert = np.where(pred_uncert == small_loss_idx)[0]
            amb_size = len(idx_uncert)
            batch_size = int(amb_size / correct_size * self.batch_size)
            if batch_size < 2:
                batch_size = 2

            if amb_size <= 2:
                dataloader_ambiguous = None
            else:
                dataloader_ambiguous = DataLoader(torch.utils.data.Subset(dataset, idx[idx_uncert]),
                                                  shuffle=True,
                                                  batch_size=batch_size,
                                                  num_workers=2,
                                                  )
            # unlabeled training??? ?????? set
            idx_uncert = np.where(pred_uncert == large_loss_idx)[0]
            incorrect_size = len(idx_uncert)
            batch_size = int(incorrect_size / correct_size * self.batch_size)
            if batch_size < 2:
                batch_size = 2
            if incorrect_size <= 2:
                dataloader_incorrect = None
            else:
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
            x_u_weak = data_u["origin_img"] # transform ???
            x_u_strong = data_u["image"] # transform ???

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

                # loss??? ?????? ?????? (unlabel??? MSE loss, label??? relabel??? cross entropy ??????)
                loss = (y_l.size(0) / (y_l.size(0) + y_r.size(0) + u_size)) * criterion_L(logit_l, y_l) + \
                       (y_r.size(0) / (y_l.size(0) + y_r.size(0) + u_size)) * soft_cross_entropy_loss(logit_r,
                                                                                                      soft_pseudo) + \
                       (u_size / (y_l.size(0) + y_r.size(0) + u_size)) * criterion_U(logit_u_weak, logit_u_strong)

                _, preds = logit_l.topk(self.topk, 1, True, True)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += torch.sum(preds == y_l.unsqueeze(1)).item()
            num_data += y_l.size(0)

        n_batches = len(loader_L)
        return total_loss / n_batches, correct / num_data


    def _train(
            self, memory_loader, optimizer, criterion
    ):
        total_loss, correct, num_data = 0.0, 0.0, 0.0

        self.model.train()
        if self.robust_type == 'DivideMix':
            self.model_2.train()
        for idx, data in enumerate(memory_loader):
            x = data["image"]
            y = data["label"]

            x = x.to(self.device)
            y = y.to(self.device)

            l, c, d = self.update_model(x, y)
            total_loss += l
            correct += c
            num_data += d

        n_batches = len(memory_loader)
        return total_loss / n_batches, correct / num_data


    # sample??? current data??? ?????? ??????
    def update_memory(self, sample):
        #print("mem imgs", len(self.memory.images), "memory_size", self.memory_size)
        if len(self.memory.images) >= self.memory_size:
            label_frequency = copy.deepcopy(self.memory.cls_count)
            label_frequency[self.exposed_classes.index(sample['label'])] += 1
            cls_to_replace = np.argmax(np.array(label_frequency))
            cand_idx = self.memory.cls_idx[cls_to_replace] # ????????? max count label??? ?????? data?????? index??? ????????????.
            print("cand_idx_len", len(cand_idx))
            mem_images = list(np.array(self.memory.images)[cand_idx])
            mem_labels = list(np.array(self.memory.labels)[cand_idx])
            dic = {'image':mem_images, 'label':mem_labels, 'original_index':cand_idx}
            cand_df = pd.DataFrame(dic)
            cls_cand_df = self.calculate_loss_and_feature(cand_df)
            cls_cand_df = self.get_cls_cos_score(cls_cand_df)
            cls_loss = cls_cand_df["loss"].to_numpy()
            cls_loss = (cls_loss - cls_loss.mean()) / cls_loss.std()
            sim_score = cls_cand_df["similarity"].to_numpy()
            sim_score = (sim_score - sim_score.mean()) / sim_score.std()
            score = (1 - self.coeff) * cls_loss + self.coeff * sim_score
            drop_idx = np.argmax(score)
            drop_idx = cand_df.iloc[drop_idx]['original_index']
            self.memory.replace_sample(sample, drop_idx)
        else:
            self.memory.replace_sample(sample)

    def online_before_task(self, cur_iter):
        self.reset_opt()
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lambda iter: 1)

    def online_after_task(self, cur_iter):
        self.reset_opt()
        self.online_memory_train(
            cur_iter=cur_iter,
            n_epoch=self.memory_epoch,
            batch_size=self.batch_size,
        )


    def update_model(self, x, y, criterion, optimizer):
        optimizer.zero_grad()
        if self.robust_type == 'DivideMix':
            self.optimizer_2.zero_grad()

        do_cutmix = self.cutmix and np.random.rand(1) < 0.5
        if do_cutmix:
            x, labels_a, labels_b, lam = cutmix_data(x=x, y=y, alpha=1.0)
            logit = self.model(x)
            loss = lam * criterion(logit, labels_a) + (1 - lam) * criterion(
                logit, labels_b
            )
            if self.robust_type in ['CoTeaching', 'DivideMix']:
                logit_2 = self.model_2(x)
                loss += lam * criterion(logit_2, labels_a) + (1 - lam) * criterion(
                    logit_2, labels_b
                )
        else:
            logit = self.model(x)
            loss = criterion(logit, y)
            if self.robust_type in ['CoTeaching', 'DivideMix']:
                logit_2 = self.model_2(x)
                loss += criterion(logit_2, y)
                if self.robust_type == 'DivideMix' and 'asymN' in self.exp_name:
                    loss += neg_entropy_loss(logit) + neg_entropy_loss(logit_2)

        _, preds = logit.topk(self.topk, 1, True, True)

        loss.backward()
        optimizer.step()
        if self.robust_type == 'DivideMix':
            self.optimizer_2.step()

        return loss.item(), torch.sum(preds == y.unsqueeze(1)).item(), y.size(0)



    def online_train(self, sample, batch_size, n_worker, iterations=1, stream_batch_size=1):
        print("dfs")
        total_loss, correct, num_data = 0.0, 0.0, 0.0
        if stream_batch_size > 0:
            sample_dataset = StreamDataset(sample, dataset=self.dataset, transform=self.train_transform,
                                           cls_list=self.exposed_classes, data_dir=self.data_dir, device=self.device,
                                           transform_on_gpu=self.gpu_transform)
        if len(self.memory) > 0 and batch_size - stream_batch_size > 0:
            memory_batch_size = min(len(self.memory), batch_size - stream_batch_size)

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
            
            l, c, d = self.update_model(x, y, self.criterion, self.optimizer)

            '''
            self.optimizer.zero_grad()
            logit, loss = self.model_forward(x,y)

            _, preds = logit.topk(self.topk, 1, True, True)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            self.update_schedule()

            total_loss += loss.item()
            correct += torch.sum(preds == y.unsqueeze(1)).item()
            num_data += y.size(0)
            '''
            total_loss += l
            correct += c
            num_data += d

        return total_loss / iterations, correct / num_data

    def evaluations(self, model, test_loader, criterion):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)
                logit = model(x)

                loss = criterion(logit, y)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)

                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.item()
                label += y.tolist()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        ret = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}

        return ret

    def online_memory_train(self, cur_iter, n_epoch, batch_size):
        if len(self.memory_list) > 0:
            mem_dataset = MyImageDataset(
                pd.DataFrame(self.memory_list),
                dataset_path=self.dataset_path,
                transform=self.train_transform,
            )
            if self.robust_type == "PuriDivER":
                mem_dataset = MyImageDataset(
                    pd.DataFrame(self.memory_list),
                    dataset_path=self.dataset_path,
                    transform=[self.weak_transform, self.train_transform, self.test_transform],
                )
                split_dataset = MyImageDataset(
                    pd.DataFrame(self.memory_list),
                    dataset_path=self.dataset_path,
                    transform=self.test_transform,
                )
            elif self.robust_type == 'DivideMix':
                mem_dataset = MyImageDataset(
                    pd.DataFrame(self.memory_list),
                    dataset_path=self.dataset_path,
                    transform=[self.weak_transform, self.weak_transform, self.test_transform],
                )
                split_dataset = MyImageDataset(
                    pd.DataFrame(self.memory_list),
                    dataset_path=self.dataset_path,
                    transform=self.test_transform,
                )
            memory_loader = DataLoader(
                mem_dataset,
                shuffle=True,
                batch_size=batch_size,
                num_workers=n_worker,
            )
            stream_batch_size = batch_size
        else:
            memory_loader = None
            stream_batch_size = batch_size

        # train_list == streamed_list in RM
        train_list = self.streamed_list
        test_list = self.test_list

        # Configuring a batch with streamed and memory data equally.
        train_loader, test_loader = self.get_dataloader(
            stream_batch_size, n_worker, train_list, test_list
        )
        logger.info(f"Streamed samples: {len(self.streamed_list)}")
        logger.info(f"In-memory samples: {len(self.memory_list)}")
        logger.info(f"Train samples: {len(train_list) + len(self.memory_list)}")
        logger.info(f"Test samples: {len(test_list)}")

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.lr
        if self.robust_type == 'DivideMix':
            for param_group in self.optimizer_2.param_groups:
                param_group["lr"] = self.lr

        # TRAIN
        best_acc = 0.0
        eval_dict = dict()
        cnt_consistency = dict()
        self.model = self.model.to(self.device)

        for epoch in range(n_epoch):
            if self.robust_type == "SELFIE" and epoch >= self.warmup:
                train_loss, train_acc, cnt_consistency = self.selfie(memory_loader=memory_loader,
                                                                     optimizer=self.optimizer, criterion=self.criterion,
                                                                     cnt_consistency=cnt_consistency)
            elif self.robust_type == "CoTeaching":
                train_loss, train_acc = self.coteaching(memory_loader=memory_loader, optimizer=self.optimizer,
                                                        r_t = 1 - min([epoch / 10 * (self.noise_rate), self.noise_rate]))
            elif self.robust_type == "DivideMix" and epoch >= self.warmup:
                label_loader, unlabel_loader = self.split_data(dataset=mem_dataset, test_dataset=split_dataset, n=2,
                                                               model=self.model_2)
                self.dividemix(epoch, self.model, self.model_2, self.optimizer, label_loader, unlabel_loader,
                               warm_up=self.warmup)
                label_loader, unlabel_loader = self.split_data(dataset=mem_dataset, test_dataset=split_dataset, n=2,
                                                               model=self.model)
                self.dividemix(epoch, self.model_2, self.model, self.optimizer_2, label_loader, unlabel_loader,
                               warm_up=self.warmup)
            elif self.robust_type == "PuriDivER" and epoch >= self.warmup:
                correct_loader, ambiguous_loader, incorrect_loader = self.puridiver_split(epoch, dataset=mem_dataset, n=2)
                if ambiguous_loader is not None and incorrect_loader is not None:
                    train_loss, train_acc = self.puridiver(correct_loader, ambiguous_loader,
                                                           incorrect_loader,
                                                           optimizer=self.optimizer)
                else:
                    train_loss, train_acc = self._train(memory_loader=memory_loader,
                                                        optimizer=self.optimizer, criterion=self.criterion)
            else:
                train_loss, train_acc = self._train(memory_loader=memory_loader,
                                                    optimizer=self.optimizer, criterion=self.criterion)
            eval_dict = self.evaluations(
                model=self.model, test_loader=test_loader, criterion=self.criterion
            )

            writer.add_scalar(f"task{cur_iter}/train/loss", train_loss, epoch)
            writer.add_scalar(f"task{cur_iter}/train/acc", train_acc, epoch)
            writer.add_scalar(f"task{cur_iter}/test/loss", eval_dict["avg_loss"], epoch)
            writer.add_scalar(f"task{cur_iter}/test/acc", eval_dict["avg_acc"], epoch)
            writer.add_scalar(f"task{cur_iter}/train/lr", self.optimizer.param_groups[0]["lr"], epoch)

            logger.info(
                f"Task {cur_iter} | Epoch {epoch + 1}/{n_epoch} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | "
                f"test_loss {eval_dict['avg_loss']:.4f} | test_acc {eval_dict['avg_acc']:.4f} | "
                f"lr {self.optimizer.param_groups[0]['lr']:.4f}"
            )

            self.scheduler.step()
            if self.robust_type == 'DivideMix':
                self.scheduler_2.step()

            best_acc = max(best_acc, eval_dict["avg_acc"])

        return best_acc, eval_dict
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
            print("idxlist", len(idxlist))
            print(idxlist)
            for idx in idxlist:
                #data = mem_dataset.get_data_gpu(idx)
                data = mem_dataset[idx]
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

'''
