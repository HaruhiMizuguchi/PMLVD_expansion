import numpy as np
import csv
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.optimize import nnls
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances

def read_data(data,p_noise,p_val,cv):
    if data in ["music_emotion","music_style"]:
        batch_size = 200
    elif data in ["mirflickr"]:
        batch_size = 500
    elif data in ['CAL500','emotions','genbase']:
        batch_size = 50
    elif data in ['scene','enron']:
        batch_size = 100

    feature = np.loadtxt("data/"+data+"/data.csv", delimiter=",")
    cand = np.loadtxt("data/"+data+"/cand/"+str(p_noise)+".csv", delimiter=",")
    ground_truth = np.loadtxt("data/"+data+"/target.csv", delimiter=",")
    cv_ind = np.loadtxt("data/"+data+"/index/5-cv.csv", delimiter=",")
    val_ind = np.loadtxt("data/"+data+"/index/true/"+str(p_val)+"/A.csv", delimiter=",")
    with open("data/"+data+"/index/true_5rows/"+str(p_val)+"/A.csv") as f:
        reader = csv.reader(f)
        l = [row for row in reader]
    val_ind_5rows = [[int(v) for v in row] for row in l]

    train_ind = np.where(cv_ind != cv)
    test_ind = np.where(cv_ind != cv)
    train_val_ind = np.array([],dtype = "int")
    for i in range(5):
        if i != cv:
            train_val_ind = np.append(train_val_ind,np.array(val_ind_5rows[i],dtype="int"))
    train_noise_ind = np.setdiff1d(train_ind,train_val_ind)

    cand_with_val = cand
    cand_with_val[train_val_ind] = ground_truth[train_val_ind]

    train_feature = feature[train_ind]
    train_cand = cand_with_val[train_ind]
    train_gt = ground_truth[train_ind]
    test_feature = feature[test_ind]
    test_gt = ground_truth[test_ind]

    train_val_feature = feature[train_val_ind]
    train_val_gt = ground_truth[train_val_ind]
    train_noise_feature = feature[train_noise_ind]
    train_noise_cand = cand[train_noise_ind]
    train_noise_gt = ground_truth[train_noise_ind]

    return(train_feature,train_cand,train_gt,test_feature,test_gt,train_val_feature,train_val_gt,train_noise_feature,train_noise_cand,train_noise_gt,batch_size)

def predict_creds(train_val_feature,train_val_gt,train_noise_feature,train_noise_cand,mode,kind_dist,k,alpha=-1):
    
    # 正規化する関数
    def regu(a):
        a = a.T
        scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
        scaler.fit(a.astype('float'))
        aa = scaler.transform(a.astype('float')).T
        return(aa)

    # 検証集合全てで線形近似
    def default_predict(train_val_feature,train_val_gt,train_noise_feature,train_noise_cand):
        # ノイズ付き集合の特徴を検証集合の特徴で線形近似
        W = np.zeros((train_noise_feature.shape[0],train_val_feature.shape[0]))
        for i in range(train_noise_feature.shape[0]):
            W[i] = nnls(train_val_feature.T,train_noise_feature[i])[0]
        
        # 重みを用いて訓練集合の信頼度を推定
        creds_pre = (train_val_gt.T @ W.T).T
        creds_pre[np.where(train_noise_cand==0)] = 0
        creds_pre = regu(creds_pre)
        return(creds_pre)

    # 検証集合とノイズ付き集合のknnでそれぞれ線形近似した後重みをつけて足す
    def predict_default_plus_knn(train_val_feature,train_val_gt,train_noise_feature,train_noise_cand,kind_dist,k,alpha):
        # 検証集合のみ
        W = np.zeros((train_noise_feature.shape[0],train_val_feature.shape[0]))
        W_cand = np.zeros((train_noise_feature.shape[0],k))
        for i in range(train_noise_feature.shape[0]):
            W[i] = nnls(train_val_feature.T,train_noise_feature[i])[0]

        # ノイズ付き集合のknn
        if kind_dist == "euclid":
            print("euclidian distance is not implemented yet")
        elif kind_dist == "cosine":
            train_cos_sim = cosine_similarity(train_noise_feature)
            meta_cos_sim = cosine_similarity(train_val_feature)
            for i in range(train_noise_feature.shape[0]):
                sorted_ind = np.argsort(train_cos_sim[i])
                cand_features_kNN = train_noise_feature[sorted_ind[1:k+1]]
                cand_plabels_kNN = train_noise_cand[sorted_ind[1:k+1]]
                W_cand[i] = nnls(cand_features_kNN.T,train_noise_feature[i])[0]
        if alpha==-1:
            cand_weight = (np.sum(train_val_gt)/train_val_gt.shape[0]) / (np.sum(train_noise_cand)/train_noise_cand.shape[0])
        else:
            cand_weight = alpha
        creds_pre = ((1-cand_weight) * (train_val_gt.T @ W.T) + cand_weight * (cand_plabels_kNN.T @ W_cand.T)).T
        creds_pre = regu(creds_pre)
        return(creds_pre)
    
    # 全ての訓練データのkNN
    def predict_all_knn(train_val_feature,train_val_gt,train_noise_feature,train_noise_cand,kind_dist,k):
        W = np.zeros((train_noise_feature.shape[0],k))
        all_feature = np.concatenate((train_noise_feature,train_val_feature),axis=0)
        all_label = np.concatenate((train_noise_cand,train_val_gt),axis=0)
        if kind_dist == "euclid":
            print("euclidian distance is not implemented yet")
        elif kind_dist == "cosine":
            train_cos_sim = cosine_similarity(all_feature)
            for i in range(train_noise_feature.shape[0]):
                sorted_ind = np.argsort(train_cos_sim[i])
                cand_features_kNN = all_feature[sorted_ind[1:k+1]]
                cand_plabels_kNN = all_label[sorted_ind[1:k+1]]
        W[i] = nnls(cand_features_kNN.T,train_noise_feature[i])[0]
        creds_pre = (cand_plabels_kNN.T @ W.T).T
        creds_pre = regu(creds_pre)
        return(creds_pre)


    if mode == 0:  
        return(default_predict(train_val_feature,train_val_gt,train_noise_feature,train_noise_cand))
    
    elif mode == 1:
        return(predict_default_plus_knn(train_val_feature,train_val_gt,train_noise_feature,train_noise_cand,kind_dist,k,alpha))
    
    elif mode == 2:
        return(predict_all_knn(train_val_feature,train_val_gt,train_noise_feature,train_noise_cand,kind_dist,k))
    
    else:
        print("this mode is not supported.\n")

def get_dataloader(train_feature,train_cand,train_gt,test_feature,test_gt,train_val_feature,train_val_gt,train_noise_feature,train_noise_cand,creds_pre,batch_size):
    
    features_num = train_feature.shape[1]
    labels_num = train_gt.shape[1]
    
    train_noise_feature_tensor = torch.tensor(train_noise_feature, dtype=torch.float)
    train_noise_cand_tensor = torch.tensor(train_noise_cand, dtype=torch.float)
    train_noise_creds_tensor = torch.tensor(creds_pre,dtype=torch.float)
    train_val_feature_tensor = torch.tensor(train_val_feature, dtype=torch.float)
    train_val_gt_tensor = torch.tensor(train_val_gt, dtype=torch.float)
    test_feature_tensor = torch.tensor(test_feature,dtype=torch.float)
    test_gt_tensor = torch.tensor(test_gt,dtype=torch.float)

    # Baseline, PML-MD用のデータセット
    train_noise_dataset = TensorDataset(train_noise_feature_tensor,train_noise_cand_tensor,train_noise_cand_tensor)
    # PML-VD用のデータセット．推定した信頼度を含む．
    train_noise_dataset_with_creds = TensorDataset(train_noise_feature_tensor,train_noise_cand_tensor,train_noise_creds_tensor)
    # 検証集合
    train_val_dataset = TensorDataset(train_val_feature_tensor,train_val_gt_tensor,train_val_gt_tensor)
    test_dataset = TensorDataset(test_feature_tensor, test_gt_tensor)

    noise_loader = DataLoader(dataset=train_noise_dataset, batch_size=batch_size,shuffle=True)
    creds_loader = DataLoader(dataset=train_noise_dataset_with_creds,batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=train_val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_feature.shape[0], shuffle=False)

    return noise_loader, creds_loader, val_loader, test_loader, features_num, labels_num