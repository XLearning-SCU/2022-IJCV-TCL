import os
import argparse
import torch
import numpy as np
from utils import yaml_config_hook
import network
import argparse
from utils import cluster_utils
from torch.utils import data
from sentence_transformers import SentenceTransformer


class DatasetIterater(data.Dataset):
    def __init__(self, texta, label):
        self.texta = texta
        self.label = label

    def __getitem__(self, item):
        return self.texta[item], self.label[item]

    def __len__(self):
        return len(self.texta)


def inference(loader, model, device):
    model.eval()
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        with torch.no_grad():
            c = model.forward_cluster(x)
        c = c.detach()
        feature_vector.extend(c.cpu().detach().numpy())
        labels = []
        for i in y:
            labels.append(int(i))
        labels_vector.extend(np.array(labels))
        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing features...")
    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    print("Features shape {}".format(feature_vector.shape))
    return feature_vector, labels_vector


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluation", add_help=False)
    parser.add_argument(
        "--dataset_dir", default="./datasets/", type=str, help="dataset path",
    )
    parser.add_argument(
        "--dataset",
        default="StackOverflow",
        type=str,
        help="dataset",
        choices=["StackOverflow", "Biomedical"],
    )
    parser.add_argument(
        "--class_num", default=20, type=int, help="number of the clusters",
    )
    parser.add_argument("--feature_dim", default=128, type=int, help="dimension of ICH")
    parser.add_argument("--eval_epoch", default=1, type=int)
    parser.add_argument("--gpu", default='1', type=str)
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--model_path",
        default="./save/StackOverflow/release-text/",
        help="path where to save, empty for no saving",
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data, label = [], []
    with open(os.path.join(args.dataset_dir, args.dataset + '.txt'), 'r') as f1:
        for line in f1:
            data.append(line.strip('\n'))
    with open(os.path.join(args.dataset_dir, args.dataset + '_gnd.txt'), 'r') as f2:
        for line in f2:
            label.append(line.strip('\n'))
    dataset = DatasetIterater(data, label)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=500,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    text_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens', device='cuda')
    class_num = args.class_num
    model = network.Network(text_model, args.feature_dim, class_num)
    model = model.to('cuda')
    model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.eval_epoch))
    model.load_state_dict(torch.load(model_fp, map_location=device.type)['net'])
    model.to(device)

    print("### Creating features from model ###")
    X, Y = inference(data_loader, model, device)
    Y = Y - 1
    print(np.min(X), np.min(Y))

    score, _ = cluster_utils.clustering_metric(Y, X, class_num)
    print('NMI = {:.4f} ARI = {:.4f} F = {:.4f} ACC = {:.4f}'
          .format(score['NMI'], score['ARI'], score['f_measure'], score['accuracy']))
