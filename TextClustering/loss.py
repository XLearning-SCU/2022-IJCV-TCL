import copy

import torch
import torch.nn as nn
import math


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, confidence_bs, class_num, temperature_ins, temperature_clu, device):
        super(ContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.confidence_bs = confidence_bs
        self.class_num = class_num
        self.temperature_ins = temperature_ins
        self.temperature_clu = temperature_clu
        self.device = device

        self.mask_ins = self.mask_correlated(batch_size)
        self.mask_clu = self.mask_correlated(class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated(self, size):
        N = 2 * size
        mask = torch.ones((N, N)).to(self.device)
        mask = mask.fill_diagonal_(0)
        for i in range(size):
            mask[i, size + i] = 0
            mask[size + i, i] = 0
        mask = mask.bool()
        return mask

    def generate_pseudo_labels(self, c, class_num):
        pseudo_label = -torch.ones(self.confidence_bs, dtype=torch.long).to(self.device)
        tmp = torch.arange(0, self.confidence_bs).to(self.device)
        with torch.no_grad():
            prediction = c.argmax(dim=1)
            confidence = c.max(dim=1).values
            pseudo_per_class = math.ceil(self.confidence_bs / class_num * 0.5)
            for i in range(class_num):
                class_idx = (prediction == i)
                confidence_class = confidence[class_idx]
                num = min(confidence_class.shape[0], pseudo_per_class)
                confident_idx = torch.argsort(-confidence_class)
                for j in range(num):
                    idx = tmp[class_idx][confident_idx[j]]
                    pseudo_label[idx] = i
        return pseudo_label


    def forward_weighted_ce(self, c_, pseudo_label, class_num):
        idx, counts = torch.unique(pseudo_label, return_counts=True)
        freq = pseudo_label.shape[0] / counts.float()
        weight = torch.ones(class_num).to(pseudo_label.device)
        weight[idx] = freq
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss_ce = criterion(c_, pseudo_label)
        return loss_ce

    def forward(self, z_i, z_j, c_i, c_j, pseudo_labels, pseudo_labels_oc):
        # Entropy Loss
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j

        # Cluster Loss
        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)
        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature_clu
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)
        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask_clu].reshape(N, -1)
        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        cluster_loss = self.criterion(logits, labels)
        cluster_loss /= N

        mask = torch.eye(self.batch_size).bool().to(self.device)
        mask = mask.float()

        contrast_count = 2
        contrast_feature = torch.cat((z_i, z_j), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature_ins)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(self.batch_size * anchor_count).view(-1, 1).to(self.device), 0)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        instance_loss = - mean_log_prob_pos
        instance_loss = instance_loss.view(anchor_count, self.batch_size).mean()

        return instance_loss, cluster_loss + ne_loss

    def forward_instance_elim(self, z_i, z_j, pseudo_labels):
        # instance loss
        invalid_index = (pseudo_labels == -1)
        mask = torch.eq(pseudo_labels.view(-1, 1),
                        pseudo_labels.view(1, -1)).to(z_i.device)
        mask[invalid_index, :] = False
        mask[:, invalid_index] = False
        mask_eye = torch.eye(self.batch_size).float().to(z_i.device)
        mask &= (~(mask_eye.bool()).to(z_i.device))
        mask = mask.float()

        contrast_count = 2
        contrast_feature = torch.cat((z_i, z_j), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature_ins)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        # mask_with_eye = mask | mask_eye.bool()
        # mask = torch.cat(mask)
        mask = mask.repeat(anchor_count, contrast_count)
        mask_eye = mask_eye.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(self.batch_size * anchor_count).view(-1, 1).to(
                z_i.device), 0)
        logits_mask *= (1 - mask)
        mask_eye = mask_eye * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask_eye * log_prob).sum(1) / mask_eye.sum(1)

        # loss
        instance_loss = -mean_log_prob_pos
        instance_loss = instance_loss.view(anchor_count,
                                           self.batch_size).mean()

        return instance_loss
