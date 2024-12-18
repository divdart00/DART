'''
***********************************************************************
DART: Diversified and Accurate Long-Tail Recommendation

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

-----------------------------------------------------
File: utils.py
- functions for dataset preparation and sequence augmentation

Version: 1.0
***********************************************************************
'''

import sys
import copy
import torch
import random
import numpy as np
from collections import defaultdict, Counter
from multiprocessing import Process, Queue
from tqdm import tqdm
import numpy as np
from tqdm import tqdm
from scipy.sparse import dok_matrix
from sklearn.cluster import KMeans
import pickle

'''
Popularity-based negative sampling for one user

input:
* train_items: item ids
* frequency: frequency of items in the dataset
* s: user sequence 

returns:
* sampled negative items
'''
def popularity_based_negative_sampling(train_items, frequency, s): 
    t = np.random.choice(train_items, p=frequency/frequency.sum())
    while t in s:
        t = np.random.choice(train_items, p=frequency/frequency.sum())
    return t

'''
increase the synthesis proportion gradually (curriculum learning)

input:
* epoch: epoch
* max_val: max value of synthesis
* sharp: control increase rate
* start_curri: epoch that starts curriculumn learning

returns:
synthesis propotion 
'''
def curriculum_function(epoch, max_val, sharp, start_curri):
	return max_val*(1-np.exp(- (epoch- start_curri)/sharp))

'''
Train and test data partition function

input:
* fname: file name of dataset

returns:
* train and test data with information of dataset
'''
def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    
    f = open('./data/'+fname+'.txt', 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum] 

'''
split head and tail items

input:
* user_train: interaction dataset
* head_proportion: ratio of head items
* dataset: name of dataset

returns:
* head_items: array of head item ids
* tail_items: array of tail tiem ids
* train_items: item ids
* frequency: interaction frequency of the train_items in the dataset
* sorted_items: sorted item id according to its frequency
'''
def head_tail_split(user_train, head_proportion, dataset):
    all_items=[]
    f = open('./data/'+dataset+'.txt', 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        i = int(i)
        all_items.append(i)
    f.close()
    item_counts = Counter(all_items)
    sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)

    train_items = np.array([item[0] for item in sorted_items])
    frequency = np.array([item[1] for item in sorted_items])

    top_20_percent_index = int(len(sorted_items) * head_proportion)
    head_items = np.array([item[0] for item in sorted_items[:top_20_percent_index]])
    tail_items=np.array([item[0] for item in sorted_items[top_20_percent_index:]])
   
    return head_items, tail_items, train_items, frequency, sorted_items

'''
Sample function of all user sequences

input:
* user_train: interaction history of each user
* usernum: number of users
* itemnum: number of items
* batch_size: size of batch
* maxlen: maximum length of user sequence
* result_queue: queue to save sampling result
* SEED: random seed
'''
def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED, train_items, frequency):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = popularity_based_negative_sampling(train_items, frequency, ts) 
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))

'''
Random Masking for robustness
'''
def random_masking(seq, general_masking_proportion, masked_items, args):
    tensor_seq=torch.LongTensor(seq).to(args.device)
    target_tensor=torch.LongTensor(masked_items).to(args.device)

    target_mask=torch.isin(tensor_seq, target_tensor).int() 

    probability_of_one = general_masking_proportion 
    probability_tensor = torch.full(size=target_mask.shape, fill_value=probability_of_one)
    tensor2 = torch.bernoulli(probability_tensor).to(args.device)
    mask=torch.mul(target_mask, tensor2.long()) 

    mask = 1 - mask
    masked_tensor=tensor_seq.mul_(mask)

    return masked_tensor

def tensor_delete(tensor, indices):
    mask = torch.ones(tensor.shape[0], dtype=torch.bool)
    mask[indices] = False
    return tensor[mask,:]

'''
Wrap Sampler to get all train sequences

input:
* user_train: interaction history of each user
* usernum: number of users
* itemnum: number of items
* batch_size: size of batch
* maxlen: maximum length of user sequence
* n_workers: number of workers to use in sampling
* alpha: aplha to control I3. Adjusted negative sampling

returns:
* user train sequences
'''
class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, train_items, frequency, args, batch_size, maxlen=10, n_workers=5):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9),
                                                      train_items,
                                                      frequency
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

'''
Evaluation of predicted results(top 10)

input:
* model: model to evaluate
* dataset: dataset ot evaluate on
* args: model details
* tailset: tail item ids 
* k, k2, k3: top k

returns:
* nDCG
* Hit Rate
* coverage
* tail coverage
* tail nDCG
* tail Hit Rate
'''
def evaluate(model, dataset, args, tailset, k, k2):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    tail_HT = 0.0
    tail_NDCG = 0.0
    
    NDCG2 = 0.0
    HT2 = 0.0
    tail_HT2 = 0.0
    tail_NDCG2 = 0.0
   
    NDCG3 = 0.0
    HT3 = 0.0
    tail_HT3 = 0.0
    tail_NDCG3 = 0.0
    

    valid_user = 0.0
    tail_valid_user = 0.0
    head_valid_user = 0.0

    total_item = []
    list_indi_tcov=[]

    total_item2 = []
    list_indi_tcov2=[]

    total_item3 = []
    list_indi_tcov3=[]

    recommended_items=[]

    users = range(1, usernum+1)

    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1: 
            continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        item_idx = list(set(range(1,itemnum+1)) - set(train[u]) - set([valid[u][0]]) | set([test[u][0]]))


        predictions = model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        _, topk2 = torch.topk(predictions, k2)
        topk2_cpu = np.array(item_idx)[topk2.cpu()]
        topk_cpu = topk2_cpu[:k]

        valid_user += 1
        if test[u][0] in tailset:
            tail_valid_user += 1
            if test[u][0] in topk_cpu:
                HT += 1
                rank = np.where(topk_cpu==test[u][0])[0]
                NDCG += 1 / np.log2(rank + 2)
                tail_HT += 1
                tail_NDCG += 1 / np.log2(rank + 2)
            if test[u][0] in topk2_cpu:
                HT2 += 1
                rank2 = np.where(topk2_cpu==test[u][0])[0]
                NDCG2 += 1 / np.log2(rank2 + 2)
                tail_HT2 += 1
                tail_NDCG2 += 1 / np.log2(rank2 + 2)

        else:
            if test[u][0] in topk_cpu:
                HT += 1
                rank = np.where(topk_cpu==test[u][0])[0]
                NDCG += 1 / np.log2(rank + 2)
                
            if test[u][0] in topk2_cpu:
                HT2 += 1
                rank2 = np.where(topk2_cpu==test[u][0])[0]
                NDCG2 += 1 / np.log2(rank2 + 2)
            

        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()  
        total_item.extend(topk_cpu)
        recommended_items.append(topk_cpu)
        total_item2.extend(topk2_cpu)

    coverage = len(Counter(total_item).keys()) / itemnum
    aggre_tailcoverage=len(set(Counter(total_item).keys()) & tailset)/len(tailset)
    coverage2 = len(Counter(total_item2).keys()) / itemnum
    aggre_tailcoverage2=len(set(Counter(total_item2).keys()) & tailset)/len(tailset)
        
    return {"NDCG@10":NDCG / valid_user, 
            "HT@10":HT / valid_user, 
            "Coverage@10": coverage, 
            "Aggre T Cov@10": aggre_tailcoverage, 
            "Tail NDCG@10":tail_NDCG / tail_valid_user, 
            "Tail HT@10": tail_HT / tail_valid_user, 
            "NDCG@20":NDCG2 / valid_user, 
            "HT@20":HT2 / valid_user, 
            "Coverage@20": coverage2, 
            "Aggre T Cov@20": aggre_tailcoverage2, 
            "Tail NDCG@20":tail_NDCG2 / tail_valid_user, 
            "Tail HT@20": tail_HT2 / tail_valid_user, 
}


'''
Cluster-based Sequence Augmentation: Head-to-Tail Replacement

input:
* seq: original sequence
* change_proportion: augmentation proportion
* head_items: araay of head item ids
* clusters: cluster information dictionary(included items)
* head_clusterid: cluster information dictionary(included head items)
* args: predefined arguments by argparse

return: 
* seq: augmented sequence
'''
def cate_based_item_changing_tensor(seq, change_proportion, head_items, clusters, head_clusterid, args):
    tensor_seq=torch.LongTensor(seq).to(args.device)
    head_items_tensor=torch.LongTensor(head_items).to(args.device)

    head_mask=torch.isin(tensor_seq, head_items_tensor).int().to(args.device)
    head_index=torch.nonzero(head_mask).to(args.device)
    change_items_num=int(change_proportion * seq.shape[1])

    n_th_remove=0
    n_th_sequence=head_index[0][0]
    remove=[]
    for index,idx in enumerate(head_index):
        if n_th_sequence == idx[0].item():
            if n_th_remove >= change_items_num:
                remove.append(index)
            else:
                n_th_remove+=1
        else:
            n_th_sequence = idx[0].item()
            n_th_remove=1
    tensor_delete(head_index, remove)  
        
    change_indices= torch.randperm(len(head_index))[:change_items_num*seq.shape[0]].to(args.device)
    extraced_index=head_index[change_indices].to(args.device)
    for one_idx in extraced_index:
        h_iid_to_change=tensor_seq[one_idx[0],one_idx[1]].item()
        corr_cluster=clusters[head_clusterid[h_iid_to_change]]
        freq=np.array(corr_cluster[2])
        tail_in_clusters=corr_cluster[1]
        inv_freq=1/freq
        if len(tail_in_clusters) > 0:
            tensor_seq[one_idx[0],one_idx[1]] = np.random.choice(tail_in_clusters, p=inv_freq/ inv_freq.sum())
    return tensor_seq

'''
extract item embeddings from the pretrained neural matrix factorization

input:
* head_items: array of head item ids
* tail_items: array of tail item ids
* args: predefined arguments by argparse

return: 
* head_embeddings: item embeddings of head items
* tail_embeddings: item embeddings of tail items
'''
def extract_item_emb(head_items, tail_items, args):
    print("nmf")
    RESULT_PATH='./result/'+ args.dataset+ '/model/nmf/'
    MF_MODEL_PATH=RESULT_PATH + args.dataset+'_nmf.pt'
    MFmodel = torch.load(MF_MODEL_PATH).to(args.device)
    new_row = torch.FloatTensor(1, MFmodel.item_emb.weight.shape[1]).uniform_(-torch.sqrt(torch.tensor(1.0 / MFmodel.item_emb.weight.shape[1])), torch.sqrt(torch.tensor(1.0 / MFmodel.item_emb.weight.shape[1]))).to(args.device)
    new_mf_embedding = torch.cat((new_row, MFmodel.item_emb.weight), dim=0)
    head_embeddings = new_mf_embedding[torch.LongTensor(head_items).to(args.device)]
    tail_embeddings = new_mf_embedding[torch.LongTensor(tail_items).to(args.device)]
    
    return head_embeddings, tail_embeddings   

'''
compute transaction time distance between items

input:
* user_train: interaction data
* itemnum: number of items
* window_co: sliding window size

return: 
* time_matrix: sparse matrix that contains time distance between items
'''
def transaction_time(user_train, itemnum, window_co):
    time_matrix = dok_matrix((itemnum+1, itemnum+1), dtype=np.float64)
    for user, items in tqdm(user_train.items()):
        for i in range(len(items)):
            current_item = items[i]
            for j in range(i+1, min(i+1+window_co, len(items))):
                next_item = items[j]
                time_matrix[current_item, next_item] += 1
                time_matrix[next_item, current_item] += 1

    for key in tqdm(time_matrix.keys()):
        value= time_matrix[key].copy()
        time_matrix[key] = 1 / value

    return time_matrix

def z_norm(tensor):
    return (tensor - tensor.mean())/ tensor.std()

'''
clustering items

input:
* user_train: interaction data
* head_items: array of head items
* tail_items: array of tail items
* sorted_items: sorted item ids
* itemnum: number of items
* num_cluster: number of clusters
* args: predefined arguments by argparse

return: 
* head_clusterid: subset of clusters that include only head item ids (key= head item ids, value: clusterid)
* clusters: full cluster dictionarys
'''
def clustering(user_train, head_items, tail_items, sorted_items, itemnum, model, num_cluster, args):
    head_embeddings, tail_embeddings = extract_item_emb(head_items, tail_items, args)
    kmeans = KMeans(n_clusters=num_cluster, init='k-means++', n_init='auto').fit(head_embeddings.detach().cpu().numpy())
    head_assignments = kmeans.labels_
    print("head cluster results")
    for i in range(10):
        print(head_assignments[i])

    clusters = {k_i: [[], [], []] for k_i in range(num_cluster)}
    # head item, tail item, tail frequency
    for i, head_item in enumerate(head_items):
        clusters[head_assignments[i]][0].append(head_item.item())

    time_matrix = transaction_time(user_train, itemnum, args.window_co)
    
    tail_assignments = []
    for tail_idx, tail_embedding in tqdm(enumerate(tail_embeddings.detach().cpu().numpy()), total=tail_embeddings.size(0)):
        distances = torch.norm(torch.tensor(kmeans.cluster_centers_) - torch.tensor(tail_embedding), dim=1)
        # time distance
        total_distance=0
        tail_cluster_time_dist=[]
        for cluster_id in range(0, num_cluster):
            cluster_head_items=clusters[cluster_id][0]
            for head_id in cluster_head_items:
                if time_matrix[tail_items[tail_idx], head_id] ==0:
                    total_distance+=1
                else:
                    total_distance+=time_matrix[tail_items[tail_idx], head_id]
            one_cluster_distance = total_distance/len(cluster_head_items)
            tail_cluster_time_dist.append(one_cluster_distance)
        distances= z_norm(distances) +  args.time_dist_weight* z_norm(torch.tensor(tail_cluster_time_dist))
        closest_cluster = torch.argmin(distances).item()
        tail_assignments.append(closest_cluster)
        
    sorted_dict=dict(sorted_items)
    for i, tail_item in enumerate(tail_items):
        clusters[tail_assignments[i]][1].append(tail_item.item())
        clusters[tail_assignments[i]][2].append(sorted_dict[tail_item])
    # key= head item ids, value: clusterid
    head_clusterid = {}
    for key, value_lists in clusters.items():
        for item in value_lists[0]:
            head_clusterid[item] = key
    
    return head_clusterid, clusters

