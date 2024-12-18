'''
***********************************************************************
DART: Diversified and Accurate Long-Tail Recommendation

This software may be used only for research evaluation purposes.
For other purposes (e.g., commercial), please contact the authors.

-----------------------------------------------------
File: main.py
- A main class for training and evaluation of DART.

Version: 1.0
***********************************************************************
'''
import torch
import random
import numpy as np
import numpy as np
import torch
import os
import time
import torch
import argparse
from tqdm import tqdm
from utils import *
from model import DART, DeepMF
import torch
import pickle

def str2bool(s):
    if s not in {"false", "true"}:
        raise ValueError("Not a valid boolean string")
    return s == "true"

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--lr", default=0.001, type=float)
parser.add_argument("--maxlen", default=200, type=int) 
parser.add_argument("--hidden_units", default=256, type=int) ######
parser.add_argument("--num_blocks", default=1, type=int) #######
parser.add_argument("--num_epochs", default=500, type=int)
parser.add_argument("--num_heads", default=1, type=int)
parser.add_argument("--dropout_rate", default=0.2, type=float)
parser.add_argument("--l2_emb", default=0.0, type=float)
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--inference_only", default=0, type=int)
parser.add_argument("--state_dict_path", default=None, type=str)
parser.add_argument("--dataset", default="books", type=str)
parser.add_argument("--train_dir", default="1", type=str) 
parser.add_argument("--GPU", default="1", type=str)
parser.add_argument("--tail_proportion", default=0.8, type=float)
# evaluation
parser.add_argument("--eval_inter", default=10, type=int)
parser.add_argument("--start_eval", default=0, type=int)
# contrastive learning
parser.add_argument("--max_cl_weight", default=0.0001, type=float) 
#curriculum 
parser.add_argument("--curri_gd", default=100, type=float) 
parser.add_argument("--cl_gd", default=100, type=float) 
parser.add_argument("--start_curri", default=0, type=float) 
#cluster
parser.add_argument("--num_cluster", default=1000, type=int)
parser.add_argument("--time_dist_weight", default=1, type=float)
parser.add_argument("--window_co", default=2, type=int)
parser.add_argument("--exist_cluster", default="yes", type=str)
# synthetic sequence
parser.add_argument("--general_masking_proportion", default=0.15, type=float)
parser.add_argument("--weak_change_proportion", default=0.1, type=float) 
parser.add_argument("--strong_change_proportion", default=0.3, type=float)
parser.add_argument("--temperature", default=1, type=float) 
parser.add_argument("--seed", default=0, type=int) 

args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU  # Set the GPUs to use

print("agrs.device", args.device)
print("torch available", torch.cuda.is_available())


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)


if __name__ == "__main__":
    if args.time_dist_weight ==1:
        args.time_dist_weight=int(args.time_dist_weight)
    
    # hyperparameter setting
    if args.dataset == "books":
        args.num_cluster = 500
        args.time_dist_weight = 1
        args.start_curri = 100
        args.strong_change_proportion = 0.3
        args.weak_change_proportion = 0.1
        args.general_masking_proportion = 0.15
        args.start_eval = 100
        # args.batch_size = 1024

    elif args.dataset == "ml":
        args.num_cluster = 200
        args.time_dist_weight = 1
        args.start_curri = 0
        args.hidden_units = 256
        args.num_blocks = 2
        args.num_heads = 2
        args.strong_change_proportion = 0.15
        args.weak_change_proportion = 0.05
        args.general_masking_proportion = 0.15
        args.dropout_rate = 0.0
        # args.batch_size = 512

    elif args.dataset == "yelp":
        args.num_cluster = 800
        args.time_dist_weight = 1
        args.start_curri = 100
        args.strong_change_proportion = 0.15
        args.weak_change_proportion = 0.05
        args.general_masking_proportion = 0.15
        args.start_eval=100
        # args.batch_size = 1024
    
    folder_dir="log/"+args.dataset + "/" + str(args.num_cluster)+"/"+ args.train_dir+"/"
    if not os.path.isdir(folder_dir):
        os.makedirs(folder_dir)
    with open(os.path.join(folder_dir, "args.txt"), "w") as f:
        f.write(
            "\n".join(
                [
                    str(k) + "," + str(v)
                    for k, v in sorted(vars(args).items(), key=lambda x: x[0])
                ]
            )
        )
    f.close()

    # read dataset
    dataset = data_partition(args.dataset)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset 
    num_batch = (
        len(user_train) // args.batch_size
    )  
    cc = 0.0
    for u in user_train:
        cc += len(user_train[u])
    print("average sequence length: %.2f" % (cc / len(user_train)))

    # split head and tail items
    h_items, t_items, train_items, frequency, sorted_items = head_tail_split(user_train, float(1-args.tail_proportion), args.dataset)
    tailset=set(t_items)

    f = open(os.path.join(folder_dir, "log.txt"), "w")
    sampler = WarpSampler(
        user_train,
        usernum,
        itemnum,
        train_items, 
        frequency,
        args,
        batch_size=args.batch_size,
        maxlen=args.maxlen,
        n_workers=12,
    )

    model = DART(usernum, itemnum, args).to(args.device) 

    # inference only without training
    if args.state_dict_path is not None:
        model.load_state_dict(
            torch.load(args.state_dict_path, map_location=torch.device(args.device))
        )
        print("load model")

    if args.inference_only:
        model.eval()
        print("inference only")
        t_test = evaluate(model, dataset, args, tailset, 10, 20)  
        print("Performance of %s" %args.dataset)      
        print(t_test)
        f.write("inference only test:"+str(t_test) + '\n')
        f.flush()

    else:
        # model parameter initialization
        for name, param in model.named_parameters():
            try:
                torch.nn.init.xavier_normal_(param.data)
            except:
                pass # just ignore those failed init layers

        model.train()
        bce_criterion = torch.nn.BCEWithLogitsLoss() 
        adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
        epoch_start_idx = 1
        T = 0.0
        t0 = time.time()
        best_performance=0.0
        early_stop=0

        # set cluster path
        cluster_dir='result/'+args.dataset+'/cluster_result/'
        if not os.path.isdir(cluster_dir):
            os.makedirs(cluster_dir)
        
        # read cluster files
        if args.exist_cluster == "yes":
            print("read cluster files")
            with open(cluster_dir+args.dataset+ '_'+ str(args.num_cluster)+'_'+ str(args.time_dist_weight)+'_'+'head_clusterid.pkl', 'rb') as t:
                head_clusterid = pickle.load(t)
            with open(cluster_dir+args.dataset+ '_'+ str(args.num_cluster)+'_'+ str(args.time_dist_weight)+'_clusters.pkl', 'rb') as a:
                clusters = pickle.load(a)

        # do clustering
        elif args.exist_cluster == "no":
            print("do clustering")
            head_clusterid, clusters = clustering(user_train, h_items, t_items, sorted_items, itemnum, model, args.num_cluster, args)
        
            with open(cluster_dir+args.dataset+ '_'+ str(args.num_cluster)+'_'+ str(args.time_dist_weight)+'_'+'head_clusterid.pkl', 'wb') as t:
                pickle.dump(head_clusterid, t)
            with open(cluster_dir+args.dataset+ '_'+ str(args.num_cluster)+'_'+ str(args.time_dist_weight)+'_clusters.pkl', 'wb') as a:
                pickle.dump(clusters, a)
            print("finish clustering")
    
    # train the model
    for epoch in tqdm(range(1, args.num_epochs + 1)):
        if args.inference_only: break 
        for step in tqdm(range(num_batch)):
            u, seq, pos, neg = sampler.next_batch() # tuples to ndarray
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)

            if epoch > args.start_curri:
                if step % 10 in [5, 8]: 
                    seq = cate_based_item_changing_tensor(seq.copy(), curriculum_function(epoch,args.strong_change_proportion,args.curri_gd, args.start_curri), h_items, clusters, head_clusterid, args).to(args.device)
                else:
                    seq=torch.LongTensor(seq).to(args.device)

                masked=random_masking(seq.copy(), curriculum_function(epoch,args.general_masking_proportion,args.curri_gd, args.start_curri), h_items, args).to(args.device)
                strong_cate = cate_based_item_changing_tensor(seq.copy(), curriculum_function(epoch,args.strong_change_proportion,args.curri_gd, args.start_curri), h_items, clusters, head_clusterid, args).to(args.device)
                weak_cate = cate_based_item_changing_tensor(seq.copy(), curriculum_function(epoch,args.weak_change_proportion,args.curri_gd, args.start_curri), h_items, clusters, head_clusterid, args).to(args.device)             
                
                pos_logits, neg_logits, masked_feats, strong_feats, weak_feats = model(u, seq, pos, neg, masked, strong_cate, weak_cate)
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
                adam_optimizer.zero_grad()
                indices = np.where(pos != 0)
                loss = model.cal_loss_3(
                    pos_logits,
                    pos_labels,
                    neg_logits,
                    neg_labels,
                    indices,
                    masked_feats,
                    strong_feats,
                    weak_feats,
                    args,
                    epoch
                )
                for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
                loss.backward()
                adam_optimizer.step()
                #print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs

            else:
                if epoch > 10 and step % 10 in [1, 9]: 
                    seq = cate_based_item_changing_tensor(seq.copy(), args.strong_change_proportion, h_items, clusters, head_clusterid, args).to(args.device)
                else:
                    seq=torch.LongTensor(seq).to(args.device)

                pos_logits, neg_logits, masked_feats, strong_feats, weak_feats= model(u, seq, pos, neg, seq, seq, seq)
                pos_labels, neg_labels = torch.ones(pos_logits.shape, device=args.device), torch.zeros(neg_logits.shape, device=args.device)
                # print("\neye ball check raw_logits:"); print(pos_logits); print(neg_logits) # check pos_logits > 0, neg_logits < 0
                adam_optimizer.zero_grad()
                
                indices = np.where(pos != 0)
                loss = bce_criterion(pos_logits[indices], pos_labels[indices])
                loss += bce_criterion(neg_logits[indices], neg_labels[indices])

                for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)
                loss.backward()
                adam_optimizer.step()
                #print("loss in epoch {} iteration {}: {}".format(epoch, step, loss.item())) # expected 0.4~0.6 after init few epochs

        # evaluate
        if  epoch > args.start_eval and epoch % args.eval_inter == 0: 
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')
            print("epoch",epoch)
            t_test = evaluate(model, dataset, args, tailset, 10, 20)          
            print(t_test)
            f.write("epoch"+str(epoch)+'\n'+"test:"+str(t_test) + '\n')
            f.flush()
            t0 = time.time()

            if t_test["Tail NDCG@10"] > best_performance:
                best_test_dict= t_test
                best_performance = t_test["Tail NDCG@10"]
                fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
                fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.maxlen)
                torch.save(model.state_dict(), os.path.join(folder_dir, fname))
                best_epoch = epoch
                early_stop=0
            else:
                early_stop+=1

            if early_stop > 20:
                t0 = time.time()
                print("early stop")
                f.flush()
                break

        model.train()
    
    if args.inference_only == 0:
        f.write('###Finish all epochs####')
        f.write("best epoch"+str(best_epoch)+'\n'+"best performance:"+str(best_test_dict) + '\n')
        f.flush()
        f.close()
        sampler.close()
    
    print("Done")
    