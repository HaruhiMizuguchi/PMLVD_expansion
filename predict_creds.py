### インスタンス数 N, 特徴数 D とすると特徴量の入力は N×D の行列
### クラス数 Q とするとラベルの入力は N×Q の行列(ラベルを持つとき1, 持たない時は0)

from function.skmultilearn import skmultilearn
import sys
sys.path.append("function/skmultilearn")
sys.path.append("function/VD_MD_Base")
from function.exec import *
from function.utils.process_data import *
from function.utils import process_data
from function.metrics.metrics_new import *

### ここを変更して実行箇所を変える
### 実行する手法
method_list = ["PML-VD","Baseline","PML-MD","PML-NI","BR","ML-kNN"]
### 実行するデータ
data_list = ['music_style','mirflickr','emotions','enron','emotions','CAL500','scene','genbase']
### マルチラベルデータセットの擬陽性ラベルの量(%)
noise_list_multi = [50,100,150]
### 検証集合サイズの訓練データに対する割合(%)
val_list = [1,3,5,7,9]
### 5分割交差検証を実行する箇所
cv_start = 0
cv_end = 5

method_list = ["PML-VD"]
data_list = ['music_emotion']
noise_list_multi = [50,100,150]
val_list = [1,3,5,7,9]
k_list = [1,2,3,4,5,6,7,8,9,10]
alpha_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
cv_start = 0
cv_end = 5

min_rloss = 5

for method in method_list:
    for data in data_list:
        if data in ['music_style','mirflickr','music_emotion']:
            noise_list = [0]
        else:
            noise_list = noise_list_multi
        for p_noise in noise_list:
            for p_val in val_list:
                
                for k in k_list:
                    for alpha in alpha_list:
                        tmp_rloss = 0
                        for cv in range(cv_start,cv_end):
                            print(f"start method={method}, data={data}, noise={str(p_noise)}, val={str(p_val)}, cv={str(cv)}")
                            ### データの読み込み
                            train_feature,train_cand,train_gt,test_feature,test_gt,train_val_feature,train_val_gt,train_noise_feature,train_noise_cand,train_noise_gt,batch_size = \
                                read_data(data,p_noise,p_val,cv)

                            ### パラメータ
                            # 信頼度の推定の方法を表す. (0,1,2)のいずれか
                            # 0：検証集合のみで特徴を線形近似
                            # 1：検証集合とノイズ付き集合のknnそれぞれで線形近似した後足し合わせる
                            # 2：全ての訓練データのknnで線形近似
                            mode = 1
                            kind_dist = "cosine" # 距離関数("euclid","cosine")のどちらか
                            """k = 3 # knnのk
                            alpha = -1"""

                            creds_pre = predict_creds(train_val_feature,train_val_gt,train_noise_feature,train_noise_cand,mode,kind_dist,k,alpha)
                            np.savetxt(f"predict_creds/plus_cand/{str(k)}/{str(alpha)}/{data}/{str(p_noise)}/{str(p_val)}/creds_{str(cv)}.csv",creds_pre,delimiter=",")
                            np.savetxt(f"predict_creds/true_target/{data}/{str(p_noise)}/{str(p_val)}/target_{str(cv)}.csv",train_noise_gt,delimiter=",")
                            tmp_rloss += rLoss(creds_pre,train_noise_gt) / 5
                            ### 評価の表示
                            print(f"data={data}, noise={str(p_noise)}, val={str(p_val)}, cv={str(cv)}, k={str(k)}, alpha={str(alpha)}")
                        if tmp_rloss < min_rloss:
                            min_rloss = tmp_rloss
                            param = f"k={str(k)},alpha={str(alpha)}"


print(f"min_rloss={str(min_rloss)},param={param}")