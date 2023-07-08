import numpy as np
import torch
import argparse
import network
import loss
from utils import save_model
from torch.utils import data
from sentence_transformers import SentenceTransformer
from EDA.augment import gen_eda
import os
import itertools
import torch.nn as nn
import nlpaug.augmenter.word as naw


def get_args_parser():
    parser = argparse.ArgumentParser("TCL for clustering", add_help=False)
    parser.add_argument(
        "--batch_size", default=128, type=int, help="Batch size per GPU"
    )
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--gpu", default='1', type=str)

    # Model parameters
    parser.add_argument("--feature_dim", default=128, type=int, help="dimension of ICH")
    parser.add_argument(
        "--instance_temperature",
        default=0.5,
        type=float,
        help="temperature of instance-level contrastive loss",
    )
    parser.add_argument(
        "--cluster_temperature",
        default=1.0,
        type=float,
        help="temperature of cluster-level contrastive loss",
    )

    # Optimizer parameters
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
    parser.add_argument(
        "--lr_backbone",
        type=float,
        default=5e-6,
        help="learning rate of backbone",
    )
    parser.add_argument(
        "--lr_head",
        type=float,
        default=5e-4,
        help="learning rate of head",
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset_dir", default="./datasets/", type=str, help="dataset path",
    )
    parser.add_argument(
        "--dataset",
        default="Biomedical",
        type=str,
        help="dataset",
        choices=["StackOverflow", "Biomedical"],
    )
    parser.add_argument(
        "--class_num", default=20, type=int, help="number of the clusters",
    )
    parser.add_argument(
        "--model_path",
        default="save/Biomedical/",
        help="path where to save, empty for no saving",
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--resume",
        default=False,
        help="resume from checkpoint",
    )
    parser.add_argument(
        "--start_epoch", default=0, type=int, help="start epoch"
    )
    parser.add_argument("--save_freq", default=1, type=int, help="saving frequency")
    parser.add_argument("--num_workers", default=10, type=int)

    return parser


class DatasetIterater(data.Dataset):
    def __init__(self, texta, textb):
        self.texta = texta
        self.textb = textb

    def __getitem__(self, item):
        return self.texta[item], self.textb[item]

    def __len__(self):
        return len(self.texta)


def train():
    loss_epoch = 0
    for step, (x_i, x_j) in enumerate(data_loader):
        optimizer.zero_grad()
        optimizer_head.zero_grad()
        z_i, z_j, c_i, c_j = model(x_i, x_j)
        c_i, c_j = nn.functional.softmax(c_i, dim=1), nn.functional.softmax(c_j, dim=1)
        loss_instance, loss_cluster = criterion.forward(z_i, z_j, c_i, c_j, None, None)
        loss = loss_cluster + loss_instance
        loss.backward()
        optimizer.step()
        optimizer_head.step()
        if step % 50 == 0:
            print(f"Step [{step}/{len(data_loader)}]\t "
                  f"loss_instance: {loss_instance.item()}\t "
                  f"loss_cluster: {loss_cluster.item()}")
        loss_epoch += loss.item()
    return loss_epoch


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["MODEL_DIR"] = '../model'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    # model and optimizer
    text_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens', device='cuda')
    class_num = args.class_num
    model = network.Network(text_model, args.feature_dim, class_num)
    model = model.to('cuda')

    optimizer = torch.optim.SGD(model.backbone.parameters(),
                                lr=args.lr_backbone,
                                weight_decay=args.weight_decay)
    optimizer_head = torch.optim.Adam(itertools.chain(model.instance_projector.parameters(),
                                                      model.cluster_projector.parameters()),
                                      lr=args.lr_head,
                                      weight_decay=args.weight_decay)

    if args.resume:
        model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.start_epoch))
        checkpoint = torch.load(model_fp)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.start_epoch = checkpoint['epoch'] + 1

    # loss
    loss_device = torch.device("cuda")
    criterion = loss.ContrastiveLoss(args.batch_size, args.batch_size, class_num, args.instance_temperature,
                                                 args.cluster_temperature, loss_device).to(loss_device)

    # pipeline
    for epoch in range(args.start_epoch, args.epochs):
        # prepare data
        data_dir = args.dataset_dir
        aug1, aug2 = [], []

        # EDA augmentation
        gen_eda(os.path.join(data_dir, args.dataset + '.txt'),
                        os.path.join(data_dir, args.dataset + 'EDA_aug.txt'),
                        0.2, 0.2, 0.2, 0.2, 1)
        with open(os.path.join(data_dir, args.dataset + 'EDA_aug.txt'), "r") as f1:
            for line in f1:
                aug1.append(line.strip('\n'))
            f1.close()

        # Roberta augmentation
        data = []
        with open(os.path.join(data_dir, args.dataset + '.txt'), "r") as f1:
            for line in f1:
                data.append(line.strip('\n'))
        aug_robert = naw.ContextualWordEmbsAug(
            model_path='roberta-base', action="substitute", device='cuda', aug_p=0.2)
        tmp = []
        internal = 400  # the larger the faster, as long as not overflow the memory
        with torch.no_grad():
            for i in range(len(data)):
                tmp.append(data[i])
                if (i + 1) % internal == 0 or (i + 1) == len(data):
                    if (i + 1) % 2000 == 0:
                        print("roberta aug: the iter {} / {}".format(i // 2000, np.ceil(len(data) / 2000)))
                    aug2.extend(aug_robert.augment(tmp))
                    tmp.clear()

        dataset = DatasetIterater(aug1, aug2)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
        )
        loss_epoch = train()

        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs:
            save_model(args, model, optimizer, optimizer_head, epoch + 1)

        print(f"Epoch [{epoch+1}/{args.epochs}]\t Loss: {loss_epoch / len(data_loader)}")
