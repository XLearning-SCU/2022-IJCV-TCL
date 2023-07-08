import numpy as np
import torch.nn as nn
import torch
from torch.nn.functional import normalize


class Network(nn.Module):
    def __init__(self, backbone, feature_dim, class_num):
        super(Network, self).__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.instance_projector = nn.Sequential(
            nn.BatchNorm1d(self.backbone.get_sentence_embedding_dimension()),
            nn.ReLU(),
            nn.Linear(self.backbone.get_sentence_embedding_dimension(),
                      self.backbone.get_sentence_embedding_dimension()),
            nn.BatchNorm1d(self.backbone.get_sentence_embedding_dimension()),
            nn.ReLU(),
            nn.Linear(self.backbone.get_sentence_embedding_dimension(), self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.BatchNorm1d(self.backbone.get_sentence_embedding_dimension()),
            nn.ReLU(),
            nn.Linear(self.backbone.get_sentence_embedding_dimension(),
                      self.backbone.get_sentence_embedding_dimension()),
            nn.BatchNorm1d(self.backbone.get_sentence_embedding_dimension()),
            nn.ReLU(),
            nn.Linear(self.backbone.get_sentence_embedding_dimension(), self.cluster_num),
        )

    def forward(self, x_i, x_j):
        h_i = self.backbone.encode(x_i, batch_size=len(x_i),
                                   convert_to_numpy=False,
                                   convert_to_tensor=True)
        h_j = self.backbone.encode(x_j, batch_size=len(x_j),
                                   convert_to_numpy=False,
                                   convert_to_tensor=True)

        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        return z_i, z_j, c_i, c_j

    def forward_c(self, x):
        h = self.backbone.encode(x, batch_size=len(x),
                                 convert_to_numpy=False,
                                 convert_to_tensor=True)
        c = self.cluster_projector(h)
        c = torch.nn.functional.softmax(c, dim=1)
        return c

    def forward_c_psd(self, x_j, pseudo_index):
        x = []
        size = len(x_j)
        for i in range(size):
            if pseudo_index[i]:
                x.append(x_j[i])
        h = self.backbone.encode(x, batch_size=len(x),
                                 convert_to_numpy=False,
                                 convert_to_tensor=True)
        c = self.cluster_projector(h)
        c = torch.nn.functional.softmax(c, dim=1)
        return c

    def forward_cluster(self, x):
        h = self.backbone.encode(x, batch_size=len(x),
                                 convert_to_numpy=False,
                                 convert_to_tensor=True)
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c
