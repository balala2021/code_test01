import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn import svm
import pickle

PATH = '/data/lijinlin/embedding_bj.pickle'
DSIC_PATH = '/data/lijinlin/disc_bj.pickle'
xiaoqu_path =''

def Emb(PATH, DISC_PATH):
    with open(PATH, 'rb') as f:
        z_np = pickle.load(f)
    with open(DISC_PATH, 'rb') as f:
        discs = pickle.load(f)
    discs = [disc for disc in discs]
    print()
    test_emb = z_np
    col_name = []
    #print(test_emb.shape)  (68969, 200)
    for i in range(test_emb.shape[1]):
        col_name.append('emb_dimension_' + str(i))

    emb_col = []
    for i in range(test_emb.shape[1]):
        emb_col.append([])
    for i in range(len(col_name)):
        for emb in test_emb:
            emb_col[i].append(emb[i])

    dic = {'小区ID': discs}
    dis_emb = pd.DataFrame(data = dic)
    for i in range(len(col_name)):
        dis_emb[col_name[i]] = emb_col[i]
    dis_emb = dis_emb.groupby('小区ID').mean()
    return dis_emb

if __name__ == '__main__':
    image_emb = Emb(PATH, DISC_PATH)
    # data
    with open(xiaoqu_path, 'rb') as f:
        xiaoqu_list = pickle.load(f)
    #里边是图片
    output = pd.merge(xiaoqu_list, image_emb, how='left', on=['小区ID'])



    