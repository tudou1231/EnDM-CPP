import copy
import random
from Bio import SeqIO
import sys
from xgboost import XGBClassifier
from sklearn import metrics
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
import time
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from train_Stank import train_my
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from statistics import median,mean
from main11 import data_d
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
#TensorFlow按需分配显存
config.allow_soft_placement = True
config.log_device_placement = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
import joblib
def me(paiming_ceshi):
    pingjun=[]
    for i in range(10):
        tong = []
        for j in range(len(paiming_ceshi)):
            tong.append(paiming_ceshi[j][i])


        tong = mean(tong)


        pingjun.append(tong)

    return round(pingjun[0],3),round(pingjun[1],3),round(pingjun[2],3),round(pingjun[3],3),round(pingjun[4],3),round(pingjun[5],3),round(pingjun[6],3),round(pingjun[7],3),round(pingjun[8],3),round(pingjun[9],3)


def pinggu(pred_res, Y_test):
    pred_label = [0 if x < 0.5 else 1 for x in pred_res]
    acc = metrics.accuracy_score(y_true=Y_test, y_pred=pred_label)
    tn, fp, fn, tp = metrics.confusion_matrix(y_pred=pred_label, y_true=Y_test).ravel()
    yan_se = tp / (tp + fn)
    yan_sp = tn / (tn + fp)
    yan_auc = metrics.roc_auc_score(y_score=pred_res, y_true=Y_test)
    yan_mcc = metrics.matthews_corrcoef(y_pred=pred_label, y_true=Y_test)
    return acc, yan_se, yan_sp, yan_mcc, yan_auc

def get_xgboost():
    xgb_model = XGBClassifier()
    return xgb_model

def get_rf():
    rf_clf = RandomForestClassifier(n_estimators=500)
    return rf_clf

def get_catboost():
    rf_clf = CatBoostClassifier(iterations=500,verbose=False)
    return rf_clf

def get_lightgbm():
    lgb_model = LGBMClassifier(n_estimators=500,max_depth=60,learning_rate=0.2,num_leavel=30)
    return lgb_model

def get_GBDT():
    kn_ = GradientBoostingClassifier()
    return kn_
def get_svm():
    clf = SVC(probability=True)
    return clf
def get_Lgst():
    lg = LogisticRegression()
    return lg
def get_de_tree():
    tree = DecisionTreeClassifier()
    return tree
def get_knn():
    kn_ = KNeighborsClassifier()
    return kn_
def get_Ada():
    ada= AdaBoostClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=200)
    return  ada
def get_Gaussian():
    kn_ = GaussianProcessClassifier()
    return kn_

def main(features,target,models,code,mach_deep_feature=False):


    std = StandardScaler()
    features = std.fit_transform(features)
     #  8
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.20,random_state=8)

    dirs = ['TextCNN', 'TextRNN', 'TextLSTM', 'Attention', 'MultiHeadAtten', 'Attention_BiLSTM', 'BiGRU_Attention',
            'CNN_LSTM', 'CNN', 'RNN', 'lstm', 'BiLSTM', 'gru', 'MLP', 'ANN']
    threshold = 0.5
    # 循环次数
    model_num = 1

    hidden_size = 30

    embedding_dimension = 100
    pool_size = 5
    fully_dimension = 128

    drop_out = 0.2
    learning_rate = 0.001
    ephch = 100

    batch_size = 32

    k_ = 5
    KF = KFold(n_splits=k_)
    acc_all = 0

    K_f= True # 设置为True的时候,每次test 数据集在第一阶段都输入进去, 第二阶段相加取平均
    jishu=1

    # 保存ACC
    yz_acc,se_all, sp_all, mcc_all, auc_all,yz_acc_ju = [], [], [], [] ,[],[]
    test_acc_all, test_se_all, test_sp_all, test_mcc_all, test_auc_all = [], [], [] ,[],[]

    for i in range(len(code)):
        yz_acc.append(0),test_acc_all.append(0)
        se_all.append(0),test_se_all.append(0)
        sp_all.append(0),test_sp_all.append(0)
        mcc_all.append(0), test_mcc_all.append(0)
        auc_all.append(0), test_auc_all.append(0)

        yz_acc_ju.append([])

    '''
    读取深度特征
    
    '''
    if mach_deep_feature == False:
        features = np.load('features/D_ALL.npy', allow_pickle=True, ).astype(np.float32)
        features = pd.read_csv(
            'G:/代码/课题方向/模型完善/试验/features/' + j + '.csv',
            header=None)

        # 标准化
        std, avgs = \
                features.std(axis=0), features.sum(axis=0) / features.shape[0]

        for i in range(features.shape[1]):  # 43 是维度, 第二维
                # print(maximums[i], minimums[i], avgs[i])
            features[:, i] = (features[:, i] - avgs[i]) / std[i]
        tr_data, te_data, tr_label, te_label = train_test_split(features, target, test_size=0.20,random_state=8)
    result=[]

    val_prob = []
    val_true = []
    test_prob = []
    test_true = []

    for train_index, test_index in KF.split(x_train, y_train):
        if mach_deep_feature == False:
            deep_X_train, deep_X_test = tr_data[train_index], tr_data[test_index]
            deep_Y_train, deep_Y_test = tr_label[train_index], tr_label[test_index]

        X_train, X_test = x_train[train_index], x_train[test_index]
        Y_train, Y_test = y_train[train_index], y_train[test_index]
        train_layer2_input = []
        test_layer2_input = []
        test_all = []
        for i in models:

            if i in dirs:
                if mach_deep_feature==False:
                    train = [np.array(deep_X_train), np.array(deep_Y_train)]
                    test = [np.array(deep_X_test), np.array(deep_Y_test)]
                    test_last=[np.array(te_data), np.array(te_label)]
                else:
                    train = [np.array(X_train), np.array(Y_train)]
                    test = [np.array(X_test), np.array(Y_test)]
                    test_last = [np.array(X_test), np.array(Y_test)]
                test.append(0.5)
                train_xgboost_res1,test_xgboost_res1, model=train_my(train, test,test_last, model_num, i, ephch, batch_size,

                         hidden_size,True)
                train_xgboost_res1 = train_xgboost_res1.reshape(len(train_xgboost_res1), 1)
                test_xgboost_res1 = test_xgboost_res1.reshape(len(test_xgboost_res1), 1)
                train_layer2_input.append(train_xgboost_res1)
                test_layer2_input.append(test_xgboost_res1)


                '''
                每一次训练集都拿去测试
                '''
                if K_f==True :
                    if mach_deep_feature == False:
                        test_a = [np.array(te_data), np.array(te_label)]
                    else:
                        test_a = [np.array(X_test), np.array(Y_test)]
                    if test_a[0].ndim != 3 :
                        length = test_a[0].shape[1]
                        test_a[0] = test_a[0].reshape(-1, 1, length)
                    pred_a = model.predict(test_a[0])[:, 1]
                    pred_a = pred_a.reshape(len(pred_a), 1)
                    test_all .append(pred_a )


            if i =="Gaussian":
                xgboost_f1  = get_Gaussian().fit(X_train, Y_train)
                train_xgboost_res1 = xgboost_f1.predict_proba(X_train)[:,1]
                train_xgboost_res1= train_xgboost_res1.reshape(len(train_xgboost_res1), 1)
                test_xgboost_res1 = xgboost_f1.predict_proba(X_test)[:,1]
                test_xgboost_res1 = test_xgboost_res1.reshape(len(test_xgboost_res1), 1)
                train_layer2_input.append(train_xgboost_res1)
                test_layer2_input.append(test_xgboost_res1)

                if K_f == True :

                    test_aa1 = xgboost_f1.predict_proba(x_test)[:, 1]
                    test_aa1 = test_aa1.reshape(len(test_aa1), 1)
                    test_all.append(test_aa1)
            if i =="XGBoost":
                xgboost_f1  = get_xgboost().fit(X_train, Y_train)
                train_xgboost_res1 = xgboost_f1.predict_proba(X_train)[:,1]
                train_xgboost_res1= train_xgboost_res1.reshape(len(train_xgboost_res1), 1)
                test_xgboost_res1 = xgboost_f1.predict_proba(X_test)[:,1]
                test_xgboost_res1 = test_xgboost_res1.reshape(len(test_xgboost_res1), 1)
                train_layer2_input.append(train_xgboost_res1)
                test_layer2_input.append(test_xgboost_res1)

                if K_f == True :

                    test_aa1 = xgboost_f1.predict_proba(x_test)[:, 1]
                    test_aa1 = test_aa1.reshape(len(test_aa1), 1)
                    test_all.append(test_aa1)
            if i =='Adaboost':
                lgst_f1 = get_Ada().fit(X_train, Y_train)
                train_lgst_res1 = lgst_f1.predict_proba(X_train)[:, 1]
                train_lgst_res1 = train_lgst_res1.reshape(len(train_lgst_res1), 1)
                test_lgst_res1 = lgst_f1.predict_proba(X_test)[:, 1]
                test_lgst_res1 = test_lgst_res1.reshape(len(test_lgst_res1), 1)
                train_layer2_input.append(train_lgst_res1)
                test_layer2_input.append(test_lgst_res1)
                if K_f == True or jishu == 10:
                    test_aa2 = lgst_f1.predict_proba(x_test)[:, 1]
                    test_aa2 = test_aa2.reshape(len(test_aa2), 1)
                    test_all.append(test_aa2)

            if i =='logistic_regression':
                lgst_f1 = get_Lgst().fit(X_train, Y_train)
                train_lgst_res1 = lgst_f1.predict_proba(X_train)[:, 1]
                train_lgst_res1 = train_lgst_res1.reshape(len(train_lgst_res1), 1)
                test_lgst_res1 = lgst_f1.predict_proba(X_test)[:, 1]
                test_lgst_res1 = test_lgst_res1.reshape(len(test_lgst_res1), 1)
                train_layer2_input.append(train_lgst_res1)
                test_layer2_input.append(test_lgst_res1)
                if K_f == True :
                    test_aa2 = lgst_f1.predict_proba(x_test)[:, 1]
                    test_aa2 = test_aa2.reshape(len(test_aa2), 1)
                    test_all.append(test_aa2)
            if i =="random_forest":
                rf_f1  = get_rf().fit(X_train, Y_train)
                train_rf_res1 = rf_f1.predict_proba(X_train)[:,1]
                test_rf_res1 = rf_f1.predict_proba(X_test)[:,1]
                train_rf_res1 = train_rf_res1.reshape(len(train_rf_res1), 1)
                test_rf_res1  = test_rf_res1 .reshape(len(test_rf_res1 ), 1)
                train_layer2_input.append(train_rf_res1)
                test_layer2_input.append(test_rf_res1)
                if K_f == True :
                    test_aa3= rf_f1.predict_proba(x_test)[:, 1]
                    test_aa3 = test_aa3.reshape(len(test_aa3), 1)
                    test_all.append(test_aa3)
            if i =='CatBoost':
                catboost_f1  = get_catboost().fit(X_train, Y_train)
                train_catboost_res1 = catboost_f1.predict_proba(X_train)[:,1]
                test_catboost_res1 = catboost_f1.predict_proba(X_test)[:,1]
                train_catboost_res1 = train_catboost_res1.reshape(len(train_catboost_res1), 1)
                test_catboost_res1 = test_catboost_res1.reshape(len(test_catboost_res1), 1)
                train_layer2_input.append(train_catboost_res1)
                test_layer2_input.append(test_catboost_res1)
                #
                if K_f == True :
                    test_aa4 = catboost_f1.predict_proba(x_test)[:, 1]

                    test_aa4 = test_aa4.reshape(len(test_aa4), 1)
                    test_all.append(test_aa4)
            if i == 'LightGBM':
                lightgbm_f1 = get_lightgbm().fit(X_train, Y_train)
                train_lightgbm_res1 = lightgbm_f1.predict_proba(X_train)[:,1]
                test_lightgbm_res1 = lightgbm_f1.predict_proba(X_test)[:,1]
                train_lightgbm_res1 = train_lightgbm_res1.reshape(len(train_lightgbm_res1), 1)
                test_lightgbm_res1 = test_lightgbm_res1.reshape(len(test_lightgbm_res1), 1)
                train_layer2_input.append(train_lightgbm_res1)
                test_layer2_input.append(test_lightgbm_res1)
                if K_f == True :
                    test_aa5 = lightgbm_f1.predict_proba(x_test)[:, 1]
                    test_aa5 = test_aa5.reshape(len(test_aa5), 1)
                    test_all.append(test_aa5)
            if i =='GBDT':
                GBDT_f1 = get_GBDT().fit(X_train, Y_train)
                train_GBDT_f1_res1 = GBDT_f1.predict_proba(X_train)[:,1]
                test_GBDT_f1_res1 = GBDT_f1.predict_proba(X_test)[:,1]
                train_GBDT_f1_res1 = train_GBDT_f1_res1.reshape(len(train_GBDT_f1_res1), 1)
                test_GBDT_f1_res1 = test_GBDT_f1_res1.reshape(len(test_GBDT_f1_res1), 1)
                train_layer2_input.append(train_GBDT_f1_res1)
                test_layer2_input.append(test_GBDT_f1_res1)
                if K_f == True :
                    test_aa6 = GBDT_f1.predict_proba(x_test)[:, 1]
                    test_aa6 = test_aa6.reshape(len(test_aa6), 1)
                    test_all.append(test_aa6)
            if i=='SVM':
                svm_f1 = get_svm().fit(X_train, Y_train)
                train_svm_f1_res1 = svm_f1.predict_proba(X_train)[:,1]
                test_svm_f1_res1 = svm_f1.predict_proba(X_test)[:,1]

                train_svm_f1_res1 = train_svm_f1_res1.reshape(len(train_svm_f1_res1), 1)
                test_svm_f1_res1 = test_svm_f1_res1.reshape(len(test_svm_f1_res1), 1)
                train_layer2_input.append(train_svm_f1_res1)
                test_layer2_input.append(test_svm_f1_res1)
                if K_f == True :
                    print('jishu:', jishu)
                    test_aa7= svm_f1.predict_proba(x_test)[:, 1]
                    test_aa7 = test_aa7.reshape(len(test_aa7), 1)
                    test_all.append(test_aa7)
            if i=='decision_tree':
                de_tree_f1 = get_de_tree().fit(X_train, Y_train)
                train_de_tree_f1_f1_res1 = de_tree_f1.predict_proba(X_train)[:,1]
                test_de_tree_f1_f1_res1 = de_tree_f1.predict_proba(X_test)[:,1]
                train_de_tree_f1_f1_res1 = train_de_tree_f1_f1_res1.reshape(len(train_de_tree_f1_f1_res1), 1)
                test_de_tree_f1_f1_res1 = test_de_tree_f1_f1_res1.reshape(len(test_de_tree_f1_f1_res1), 1)
                train_layer2_input.append(train_de_tree_f1_f1_res1)
                test_layer2_input.append(test_de_tree_f1_f1_res1)
                if K_f == True :
                    test_aa8 = de_tree_f1.predict_proba(x_test)[:, 1]
                    test_aa8 = test_aa8.reshape(len(test_aa8), 1)
                    test_all.append(test_aa8)

            if i=='KNN':
                de_knn = get_knn().fit(X_train, Y_train)
                train_de_tree_f1_f1_res1 = de_knn.predict_proba(X_train)[:,1]
                test_de_tree_f1_f1_res1 = de_knn.predict_proba(X_test)[:,1]
                train_de_tree_f1_f1_res1 = train_de_tree_f1_f1_res1.reshape(len(train_de_tree_f1_f1_res1), 1)
                test_de_tree_f1_f1_res1 = test_de_tree_f1_f1_res1.reshape(len(test_de_tree_f1_f1_res1), 1)
                train_layer2_input.append(train_de_tree_f1_f1_res1)
                test_layer2_input.append(test_de_tree_f1_f1_res1)
                if K_f == True :
                    test_aa8 = de_knn.predict_proba(x_test)[:, 1]
                    test_aa8 = test_aa8.reshape(len(test_aa8), 1)
                    test_all.append(test_aa8)




        train_layer2_inputs = np.concatenate([train_layer2_input[0], train_layer2_input[1]], axis=-1)
        for i in range(2,len(train_layer2_input)):
            train_layer2_inputs=np.concatenate([train_layer2_inputs,train_layer2_input[i]],axis=-1)

        test_layer2_inputs = np.concatenate([test_layer2_input[0], test_layer2_input[1]], axis=-1)
        for i in range(2,len(test_layer2_input)):
            test_layer2_inputs=np.concatenate([test_layer2_inputs,test_layer2_input[i]],axis=-1)
        test_alls = np.concatenate([test_all[0], test_all[1]], axis=-1)
        for i in range(2, len(test_all)):
            test_alls = np.concatenate([test_alls, test_all[i]], axis=-1)

        stdd = StandardScaler()
        train_layer2_inputs = stdd.fit_transform(train_layer2_inputs)
        test_layer2_inputs = stdd.transform(test_layer2_inputs)
        test_allss = stdd.transform(test_alls)









        for i,last in enumerate(code):

                # 最后的模型预测
            if last == 'XGBoost':
                clf = get_xgboost().fit(train_layer2_inputs,Y_train)
            elif last=='random_forest':
                clf = get_rf().fit(train_layer2_inputs, Y_train)
            elif last=='CatBoost':
                clf = get_catboost().fit(train_layer2_inputs, Y_train)
            elif last=='LightGBM':
                clf = get_lightgbm().fit(train_layer2_inputs, Y_train)
            elif last=='GBDT':
                clf = get_GBDT().fit(train_layer2_inputs, Y_train)
            elif last=='SVM':
                clf = get_svm().fit(train_layer2_inputs,Y_train)
            elif last=='logistic_regression':
                clf = get_Lgst().fit(train_layer2_inputs, Y_train)
            elif last=='decision_tree':
                clf = get_de_tree().fit(train_layer2_inputs,Y_train)
            elif last=='KNN':
                clf = get_knn().fit(train_layer2_inputs,Y_train)




            pred_res = clf.predict_proba(test_layer2_inputs)[:, 1]
            val_prob.append(pred_res)
            val_true.append(Y_test)

            # 验证集的检测
            acc, yan_se, yan_sp, yan_mcc, yan_auc=pinggu(pred_res,Y_test)
            # pred_label = [0 if x < 0.5 else 1 for x in pred_res]
            # acc = metrics.accuracy_score(y_true=Y_test, y_pred=pred_label)
            # tn, fp, fn, tp = metrics.confusion_matrix(y_pred=pred_label, y_true=Y_test).ravel()
            # yan_se = tp / (tp + fn)
            # yan_sp = tn / (tn + fp)
            # yan_auc = metrics.roc_auc_score(y_score=pred_res, y_true=Y_test)
            # yan_mcc = metrics.matthews_corrcoef(y_pred=pred_label, y_true=Y_test)
            yz_acc[i]+=acc
            se_all[i] += yan_se
            sp_all[i] +=  yan_sp
            mcc_all[i]+=yan_mcc
            auc_all[i]+=yan_auc

            yz_acc_ju[i].append(round(acc,4))


            # 测试集评估
            pred_res = clf.predict_proba(test_allss)[:, 1]
            test_prob.append(pred_res)
            test_true.append(y_test)
            test_acc, test_se, test_sp, test_mcc, test_auc = pinggu(pred_res,y_test)
            test_acc_all[i]+=test_acc
            test_se_all[i]+=test_se
            test_sp_all[i]+=test_sp
            test_mcc_all[i]+=test_mcc
            test_auc_all[i]+=test_auc
            print('第{}次验证集acc:{},测试集acc:{},最终为:{},'.format(jishu, acc,test_acc, last))



            if jishu==k_:


                result.append([yz_acc[i]/k_,se_all[i]/k_,sp_all[i]/k_,mcc_all[i]/k_,auc_all[i]/k_,test_acc_all[i]/k_, test_se_all[i]/k_, test_sp_all[i]/k_,test_mcc_all[i]/k_,test_auc_all[i]/k_,yz_acc_ju[i]])

        jishu += 1


    val_prob = np.concatenate(val_prob)
    # val_ture = np.concatenate(val_true)
    test_prob = np.concatenate(test_prob)
    # test_true = np.concatenate(test_true)
    namee=''
    for i in models:
        if namee!='':
            namee=namee+'+'+i
        else:
            namee+=i

    name = 'G:/代码/课题方向/模型完善/试验/Evaluation/recurrent/'+namee + '_'
    np.save(name + 'val_prob.npy', val_prob)
    # np.save(name + 'val_true.npy', val_ture)
    np.save(name + 'test_prob.npy', test_prob)
    # np.save(name + 'test_true.npy', test_true)


    return result


if __name__=='__main__':
    columns = ['基础模型','最终', '特征', 'ACC(验证)', 'Se(验证)', 'Sp(验证)', 'MCC(验证)', 'AUC(验证)', 'ACC','Se', 'Sp', 'MCC', 'AUC','消耗时间','验证集acc合集']

    df = pd.DataFrame(data=[['-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-','-','-', '-','-',]], columns=columns)
    target = pd.read_csv('G:/代码/课题方向/模型完善/试验/features/my.csv', header=None)
    target = target.values.ravel()




    # feature = ['AAC+CTDC+QSOrder+PAAC', 'AAC+CTDC+QSOrder+PAAC+APAAC',
    #            'AAC+CTDC+QSOrder+PAAC+APAAC+EAAC', 'AAC+CTDC+QSOrder+PAAC+APAAC+EAAC+CKSAAP'
    #     , 'AAC+CTDC+QSOrder+PAAC+APAAC+EAAC+CKSAAP+DDE', 'AAC+CTDC+QSOrder+PAAC+APAAC+EAAC+CKSAAP+DDE+CTDT','AAC+CTDC+QSOrder+PAAC+APAAC+EAAC+CKSAAP+DDE+CTDT+GDPC',
    #            'AAC+CTDC+QSOrder+PAAC+APAAC+EAAC+CKSAAP+DDE+CTDT+GDPC+CTDD']
    # feature=['k_mer+peptide_select+AAC+CTDC+QSOrder+PAAC' ,]
    # feature = ['AAC+CTDC+QSOrder+PAAC', 'AAC+CTDC+QSOrder+PAAC+APAAC',]
    # feature = ['StackCPPred',] # StackCPPred  合成:'RECM-component','RCEM_DWT','PsePSSM'


    # feature = ['CPSR','Hybrid_PseAAC','KSC_20','CTDC','QSOrder','PAAC','APAAC','EAAC','CKSAAP','DDE','CTDT','GDPC','peptide_select']





    # feature = ['CPSR+peptide_select+Hybrid_PseAAC+KSC_20+CTDC+QSOrder+PAAC+APAAC+EAAC','CPSR+peptide_select+Hybrid_PseAAC+KSC_20+CTDC+QSOrder+PAAC+APAAC+EAACMRMR_1111',
    #            'CPSR+peptide_select+Hybrid_PseAAC+KSC_20+CTDC+QSOrder+PAAC+APAAC+EAAC+MRMR_1000']
    # feature =['CPSR+peptide_select+Hybrid_PseAAC+KSC_20+CTDC+QSOrder+PAAC+APAAC+EAACMRMR_1111',]




    feature = ['ML_ALL']


    # feature=['StackCPPred']
    # 基准模型
    dirs = ['RNN', 'lstm', 'BiLSTM', 'gru', 'CNN_LSTM', 'CNN', 'TextCNN', 'Attention', 'MultiHeadAtten',
            'Attention_BiLSTM', 'BiGRU_Attention', 'MPMABP', 'transformer','ANN' ]
    jizhun = ['XGBoost', 'random_forest', 'LightGBM','CatBoost','GBDT','SVM','logistic_regression','decision_tree','KNN','Gaussian']





    jizhuns = [
      ['TextCNN', 'CNN', 'SVM','random_forest',],
               ]
    jizhuns = [['TextCNN', 'CNN', 'SVM', 'Catboost',],['TextCNN', 'CNN', 'SVM', 'LightGBM',],['TextCNN', 'CNN', 'SVM', 'Catboost','LightGBM',],['TextCNN', 'CNN', 'Catboost','LightGBM',]]
    jizhuns=[ ['random_forest', 'SVM','KNN','LightGBM','XGBoost'],]  # StackCPPred
    jizhuns = [['TextCNN', 'CNN', 'SVM', 'LightGBM', ], ]
    # jizhuns = [['random_forest', 'SVM', 'KNN', 'LightGBM','XGBoost' ], ]

    # jizhuns =[['TextCNN', 'CNN','SVM','KNN','LightGBM'],]



    # 最终模型
    code=['XGBoost', 'random_forest', 'LightGBM','Catboost','GBDT','SVM','logistic_regression','decision_tree','KNN',]
    code = ['logistic_regression']
    # code = ['KNN']
    # code = ['SVM']

    # code = ['SVM',]
    # 循环多少次



    index=1

    # 遍历最后的模型
    def to_log(log):
        with open("modelLog_改.log", "a+") as f:
            f.write(log + '\n')


    for jizhun in jizhuns:

        for j in feature:
            paiming = []
            # 读取数据
            features = pd.read_csv(
                'G:/代码/课题方向/模型完善/试验/features/'+j+'.csv',
                header=None)

            time_all = 0


            start = time.time()
            print('\n---------------------\n')
            jizhun_name = ' + '.join(jizhun)
            print('基准模型{}'.format(jizhun_name))
            print('\n', " 机器特征:{}  ".format(j,))
            # 模型训练
            cmd = 'main(features,target,jizhun,code)'

            # 模型训练
            result= eval(cmd)
            #计算消耗时间
            end = time.time()
            time_all = time_all + end - start

            # 选出平均值保存
            for s,z in enumerate(code):
                # if round(result[s][0],4)>0.930:
                    # qq='基准:{},最终{},特征:{},\n,ACC(验证):{:.4f},Se(验证):{:.4f}, Sp(验证):{:.4f}, MCC(验证):{:.4f}, AUC(验证):{:.4f}, ACC:{:.4f},Se:{:.4f}, Sp:{:.4f}, MCC:{:.4f}, AUC:{:.4f}\n'\
                    #     .format(jizhun_name,z, j,result[s][0],result[s][1],result[s][2],result[s][3],result[s][4],result[s][5],result[s][6],result[s][7],result[s][8],result[s][9],)
                    # to_log(qq)

                paiming.append([jizhun_name,z, j,round(result[s][0],4),round(result[s][1],4),round(result[s][2],4),round(result[s][3],4),round(result[s][4],4),round(result[s][5],4),round(result[s][6],4),round(result[s][7],4),
                                round(result[s][8],4),round(result[s][9],4),round(time_all,2),result[s][10],])
            # 特征之间以准确率为基准排序
            paiming = sorted(paiming, key=lambda x: x[3], reverse=True)
            print(paiming)

            # # 统计特征


            for i in range(len(paiming)):
                df.loc[str(index)] = paiming[i]
                index += 1
            df.loc[str(index)] = ['  ', '  ', '  ', '  ', '  ','  ','  ','  ','  ','  ',' ',' ',' ', '-','-',]
            index += 1
            df.loc[str(index)] = ['基础模型','最终', '特征','ACC(验证)', 'Se(验证)', 'Sp(验证)', 'MCC(验证)', 'AUC(验证)', 'ACC','Se', 'Sp', 'MCC', 'AUC','消耗时间','验证集acc合集']
            index += 1

        print(df)
        name=1
        while True:
            name_zu='./excel/Stank/'
            name_zhong=name_zu+str(name)+'.xlsx'
            if os.path.exists(name_zhong) ==False:
                df.to_excel(name_zhong, engine='openpyxl',sheet_name='Sheet1',header='模型')
                break
            else:
                name+=1

        df = pd.DataFrame(data=[['  ', '  ', '  ', '  ', '  ','  ','  ','  ','  ','  ',' ',' ',' ', '-','-',]], columns=columns)