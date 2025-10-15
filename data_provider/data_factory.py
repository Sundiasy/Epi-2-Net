from data_provider.data_loader import Dataset_ETT_hour, Dataset_Custom, Dataset_M4, Dataset_Solar, Dataset_TSF, Dataset_TSF_ICL, Dataset_EpiGNN
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import math
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, asin

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'Solar': Dataset_Solar,
    'tsf': Dataset_TSF,
    'tsf_icl': Dataset_TSF_ICL,
    'epidataset': Dataset_EpiGNN
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size 
    elif flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size 
    else:
        shuffle_flag = False
        # drop_last = args.drop_last
        batch_size = args.batch_size

    if flag in ['train', 'val']:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.token_len],
            # seasonal_patterns=args.seasonal_patterns,
            # drop_short=args.drop_short,
        )
    else:
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.test_seq_len, args.test_label_len,args.test_pred_len,args.token_len],
            # seasonal_patterns=args.seasonal_patterns,
            # drop_short=args.drop_short,
        )
    if (args.use_multi_gpu and args.local_rank == 0) or not args.use_multi_gpu:
        print(flag, len(data_set))
    if args.use_multi_gpu:
        train_datasampler = DistributedSampler(data_set, shuffle=shuffle_flag)
        data_loader = DataLoader(data_set, 
            batch_size=batch_size,
            sampler=train_datasampler,
            num_workers=args.num_workers,
            persistent_workers=True,
            pin_memory=True,
            # drop_last=drop_last,
            )
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            # drop_last=drop_last
            )
    return data_set, data_loader,data_set.scaler

def haversine(lon1, lat1, lon2, lat2):
    """
    计算两个经纬度坐标之间的Haversine距离
    """
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])  # 转为弧度
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球半径（公里）
    return c * r

def get_adj(country):
    # 1. 使用DataFrame读取CSV文件
    if country == 'Italy':
        df = pd.read_csv('./dataset/Italy/italy_regions.csv')  # 确保文件路径正确
    elif country == 'Spain':
        df = pd.read_csv('./dataset/Spain/spain_regions.csv')  # 确保文件路径正确
    elif country == 'England':
        df = pd.read_csv('./dataset/England/england_regions.csv')
    elif country == 'France':
        df = pd.read_csv('./dataset/France/france_regions.csv')
        
    # 2. 创建距离矩阵
    n = len(df)
    distance_matrix = np.zeros((n, n))

    # 3. 计算两两之间的距离
    for i in range(n):
        for j in range(n):
            if i == j:
                distance_matrix[i][j] = 0  # 对角线设为0
            else:
                loc1 = df.iloc[i]
                loc2 = df.iloc[j]
                distance_matrix[i, j] = haversine(
                loc1["longitude"], loc1["latitude"],
                loc2["longitude"], loc2["latitude"]
                )
    
    # (1) 距离→相似度（使用倒数变换，避免除零）
    epsilon = 1e-6  # 防止除以0
    adj = 1 / (1 + distance_matrix + epsilon)

    # (2) 添加自环（对角线=1）
    adj = adj + np.eye(n)
    adj[:] = adj[:]/adj[:].max()  # 归一化到[0, 1]

    # (3) GNN适用的对称归一化：D^{-1/2} A D^{-1/2}
    degree_matrix = np.diag(np.sum(adj, axis=1))  # 计算度矩阵
    degree_matrix_inv_sqrt = np.linalg.inv(np.sqrt(degree_matrix))  # D^{-1/2}

    # 对称归一化
    normalized_adj = degree_matrix_inv_sqrt @ adj @ degree_matrix_inv_sqrt
    # print(normalized_adj)
    return normalized_adj