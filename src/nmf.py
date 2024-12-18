'''
***********************************************************************
DART: Diversified and Accurate Long-Tail Recommendation

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

-----------------------------------------------------
File: nmf.py
- train neural matrix factorization and get the trained model

Version: 1.0
***********************************************************************
'''
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import time
import math
import argparse
import pickle
import os
import sys
import scipy.sparse as sp
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--GPU", default="1", type=str)
parser.add_argument("--early_stop_criterion", default=10, type=int)
parser.add_argument("--test_num_ns", default=50, type=int)
parser.add_argument("--train_num_ns", default=50, type=int)
parser.add_argument("--dataset", default="books", type=str)
parser.add_argument("--seed", type=int, default=10, help="Seed")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--decay", type=float, default=1e-5, help="decay")
parser.add_argument("--batch_size", type=int, default=256, help="batch size for training")
parser.add_argument("--epochs", type=int, default=50, help="training epoches")
parser.add_argument("--factor_num", type=int, default=32, help="latent factor number")
parser.add_argument("--layers", nargs='+', default=[64, 32, 16], type=int, 
    help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU  # Set the GPUs to use
DEVICE = torch.device('cuda') 

rating_file = './data/' + args.dataset+'.txt'
RESULT_PATH='./result/'+ args.dataset+ '/model/nmf/'

if not os.path.isdir(RESULT_PATH):
    os.makedirs(RESULT_PATH)
    print("make dir success")

with open(os.path.join(RESULT_PATH, "args.txt"), "w") as f:
    f.write(
        "\n".join(
            [
                str(k) + "," + str(v)
                for k, v in sorted(vars(args).items(), key=lambda x: x[0])
            ]
        )
    )
    print("args.txt success")

'''
split train and test dataset

input:
* data: interaction data 

returns:
* train_data: train data
* test_data: test data
* train_matrix: matrix form of train data
* user_num: number of users
* item_num: number of items
'''
def train_test_split(data):

    unique_user_ids = np.unique(data[:, 0])
    test_indices = []
    for user_id in tqdm(unique_user_ids):
        user_indices = np.where(data[:, 0] == user_id)[0]
        test_indices.append(user_indices[-1])

    test_data = data[test_indices]

    print("train indicies")
    train_indices = [i for i in tqdm(range(data.shape[0])) if i not in test_indices]
    train_data = data[train_indices]
    user_num=max(data[:,0])+1
    item_num=max(data[:,1])+1
    train_matrix= sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in tqdm(train_data):
        train_matrix[x[0], x[1]] =1.0
    
    return train_data, test_data, train_matrix, user_num, item_num

# dataset class
class MovieLensDataset(Dataset): #x= (user, item), y = rating
    def __init__(self, x, y, num_item, train_matrix, num_ns, is_training): 
        super(MovieLensDataset, self).__init__()
        self.positive_samples = x # [[user, item],[user2,item2]]
        self.num_item=num_item
        self.num_ns=num_ns
        self.is_training=is_training
        self.train_matrix = train_matrix
        self.labels=y
    
    # sample negative items for training
    def set_negative_samples(self):
        assert self.is_training, "no need to sampling when testing"

        self.negative_samples=[]
        for interaction in self.positive_samples:
            user = interaction[0]
            for _ in range(self.num_ns):
                neg_item = np.random.randint(self.num_item)
                while (user, neg_item) in self.train_matrix:
                    neg_item = np.random.randint(self.num_item)
                self.negative_samples.append([user,neg_item])
        
        self.negative_samples = np.array(self.negative_samples, dtype=float)
        self.negative_samples = self.negative_samples.astype('int64')

        labels_positive = [[1]] * len(self.positive_samples)
        labels_negative = [[0]] * len(self.negative_samples)

        self.total_interactions = np.array(self.positive_samples.tolist() + self.negative_samples.tolist())

        self.total_lables = np.array(labels_positive + labels_negative)
            

    def __len__(self): 
        if self.is_training:
            return len(self.positive_samples) + self.num_ns
        else:
            return len(self.positive_samples)


    def __getitem__(self, idx):
        if self.is_training:
            features= self.total_interactions
            labels_=self.total_lables
        else:
            features= self.positive_samples
            labels_=self.labels
        one_feature = torch.LongTensor(features[idx, :])
        one_label = torch.FloatTensor(labels_[idx, :])
        return one_feature, one_label

'''
Neural Matrix Factorization model

input:
* num_users: number of users
* num_items: number of items
* args: pre-defined arguments using argparse
'''
class DeepMF(torch.nn.Module):
    def __init__(self, num_users, num_items, args):
        super(DeepMF, self).__init__()
        self.user_emb = torch.nn.Embedding(num_users, args.factor_num)
        self.item_emb = torch.nn.Embedding(num_items, args.factor_num)
        self.activation = torch.nn.ReLU()
        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(args.layers[:-1], args.layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))
            self.fc_layers.append(torch.nn.ReLU())
        self.affine_output = torch.nn.Linear(in_features=args.layers[-1], out_features=1)

    def forward(self, user_idx, item_idx):
        out = torch.cat((self.user_emb(user_idx), self.item_emb(item_idx)), dim=1)
        for idx, _ in enumerate(range(len(self.fc_layers))):
            out = self.activation(self.fc_layers[idx](out))

        out = self.activation(self.affine_output(out))
        return out    

'''
initialize model parameters

input:
* m: target model
'''
def initialize_weights(m):
    if isinstance(m, torch.nn.Embedding):
        torch.nn.init.normal_(m.weight.data, std=0.01)

    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0)

'''
check hit or not

input:
* gt_item: ground truth item
* pred_items: recommended items
'''
def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0

'''
calculate ndcg

input:
* gt_item: ground truth item
* pred_items: recommended items
'''
def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    return 0

'''
calculate metrics

input:
* model: model
* test_loader: data loader for test 
* top_k: top k number

return:
* np.mean(HR): hit ratio
* np.mean(NDCG): nDCG
'''
def metrics(model, test_loader, top_k):
    print("evaluating")
    HR, NDCG = [], []
    for batch_idx, (features, label) in enumerate(test_dataloader):
        features, label = features.to(DEVICE), label.to(DEVICE)
        users, gt_items = features[:, 0], features[:, 1] # user id, item id
        for idx in range(len(features)):
            neg_samples=[]
            for _ in range(args.test_num_ns):
                test_neg_item = np.random.randint(item_num)
                while (users[idx], test_neg_item) in train_matrix:
                    test_neg_item = np.random.randint(item_num)
                neg_samples.append([users[idx],test_neg_item])
            item_idx= torch.cat((torch.unsqueeze(features[idx], dim=0), torch.LongTensor(neg_samples).to(DEVICE)))
            pred = model(item_idx[:,0], item_idx[:,1])
            _, indices = torch.topk(pred.reshape(-1), k=top_k)
            recommends = torch.take(item_idx[:,1], indices).cpu().numpy().tolist()
            gt_item = item_idx[:,1][0].item()
            HR.append(hit(gt_item, recommends))
            NDCG.append(ndcg(gt_item, recommends))

    return np.mean(HR), np.mean(NDCG)

if __name__ == "__main__":
    # read dataset
    raw = []
    with open(rating_file, 'r') as f_read:
        for line in f_read.readlines()[0:]:
            line_list = line.strip().split(' ')
            raw.append(line_list)
    for i, row in enumerate(raw):
        row.append(2)  
    raw = np.array(raw, dtype=float)
    raw = raw.astype('int64')
    raw= raw-1
    print("data load success")

    # split train and test dataset
    train, test, train_matrix, user_num, item_num = train_test_split(raw[:,:])

    with open(RESULT_PATH + args.dataset+'_train.pkl', 'wb') as f:
        pickle.dump(train, f)
    with open(RESULT_PATH + args.dataset+'_test.pkl', 'wb') as f:
        pickle.dump(test, f)
    with open(RESULT_PATH + args.dataset+'_train_matrix.pkl', 'wb') as f:
        pickle.dump(train_matrix, f)
    print("save pickle")

    # construct dataset 
    train_dataset = MovieLensDataset(train[:, :-1], np.expand_dims(train[:, -1], axis=1), item_num, train_matrix, num_ns=args.train_num_ns, is_training=True) 
    print("dataset create")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    print("train dataloader success")
    test_dataset = MovieLensDataset(test[:, :-1], np.expand_dims(test[:, -1], axis=1), item_num, train_matrix, num_ns=0, is_training=False)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, pin_memory=True)
    print("test data loader success")
    train_dataloader.dataset.set_negative_samples()

    # model setting
    model = DeepMF(user_num, item_num, args) 
    model.apply(initialize_weights)
    model.to(DEVICE)
    print("model setting success")

    # loss and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    # train and save model
    best_performance=-10
    early_stop=0
    for epoch in tqdm(range(args.epochs)):
        start_time = time.time()
        train_mse = 0.
        test_mse = 0.

        model.train()
        for batch_idx, (features, label) in tqdm(enumerate(train_dataloader)):
            # get data
            features, label = features.to(DEVICE), label.to(DEVICE)
            user, item = features[:, 0], features[:, 1] # user id, item id

            # set gradients to zero
            optimizer.zero_grad()

            # predict ratings
            pred = model(user, item)

            loss = criterion(pred, label)
            #print("loss:",loss.item())
            loss.backward()
            optimizer.step()


        model.eval()
        HR, NDCG = metrics(model, test_dataloader, top_k=10)
            
        elapsed_time = time.time() - start_time
        
        print(
                "The time elapse of epoch {:03d}".format(epoch)
                + " is: "
                + time.strftime("%H: %M: %S", time.gmtime(elapsed_time))
            )
        print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))
        
        if HR > best_performance:
            best_performance = HR
            torch.save(model, RESULT_PATH + args.dataset+'_nmf.pt')
        else:
            early_stop+=1
        
        sys.stdout.flush()
        if early_stop >= args.early_stop_criterion:
            print("Early Stop")
            print("Best performance", HR)
            break