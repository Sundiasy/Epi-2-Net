import os
import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from data_provider.m4 import M4Dataset, M4Meta
# from sklearn.preprocessing import StandardScaler
from utils.tools import convert_tsf_to_dataframe
import warnings
from datetime import date, timedelta
warnings.filterwarnings('ignore')
import networkx as nx

def generate_new_features(Gs, labels, dates, window=7, scaled=False):
    """
    Generate node features
    Features[1] contains the features corresponding to y[1]
    e.g. if window = 7, features[7]= day0:day6, y[7] = day7
    if the window reaches before 0, everything is 0, so features[3] = [0,0,0,0,day0,day1,day2], y[3] = day3
    """
    features = list()
    
    labs = labels.copy()
    nodes = Gs[0].nodes()
  

    #--- one hot encoded the region
    
    for idx,G in enumerate(Gs):
        #  Features = population, coordinates, d past cases, one hot region
        
        H = np.zeros([G.number_of_nodes(),window]) #+3+n_departments])#])#])
        me = labs.loc[:, dates[:(idx)]].mean(1)
        sd = labs.loc[:, dates[:(idx)]].std(1)+1

        ### enumarate because H[i] and labs[node] are not aligned
        for i,node in enumerate(G.nodes()):
            #---- Past cases      
            if(idx < window):# idx-1 goes before the start of the labels
                if(scaled):
                    #me = np.mean(labs.loc[node, dates[0:(idx)]]
                    H[i,(window-idx):(window)] = (labs.loc[node, dates[1:(idx+1)]] - me[node])/ sd[node]
                else:
                    H[i,(window-idx):(window)] = labs.loc[node, dates[1:(idx+1)]]

            elif idx >= window:
                if(scaled):
                    H[i,0:(window)] =  (labs.loc[node, dates[(idx-window+1):(idx+1)]] - me[node])/ sd[node]
                else:
                    H[i,0:(window)] = labs.loc[node, dates[(idx-window+1):(idx+1)]]
      
            
        features.append(H)
        
    return features



def generate_graphs_tmp(dates, country):
    Gs = []
    for date in dates:
        # 基于国家加载数据
        if country == "IT":
            d = pd.read_csv("./dataset/Italy/graphs/"+country+"_"+date+".csv", header=None)
        elif country == "EN":
            d = pd.read_csv("./dataset/England/graphs/"+country+"_"+date+".csv", header=None)
        elif country == "FR":
            d = pd.read_csv("./dataset/France/graphs/"+country+"_"+date+".csv", header=None)
        elif country == "ES":
            d = pd.read_csv("./dataset/Spain/graphs/"+country+"_"+date+".csv", header=None)
        
        G = nx.DiGraph()
        nodes = set(d[0].unique()).union(set(d[1].unique()))
        nodes = sorted(nodes)
        G.add_nodes_from(nodes)
        # print(nodes)
        # exit()
        # 从CSV添加边及其原始权重
        edge_weights = []
        for row in d.iterrows():
            G.add_edge(row[1][0], row[1][1], weight=row[1][2])
            edge_weights.append(row[1][2])
        
        # # 计算标准化的统计数据
        # if edge_weights:
        #     min_weight = min(edge_weights)
        #     max_weight = max(edge_weights)
            
        #     # 对CSV中的权重进行Min-Max缩放到[0.1, 1.0]范围
        #     # 这确保了重要连接（来自数据）和默认连接之间有良好的区分
        #     if max_weight > min_weight:
        #         for u, v, w in G.edges(data='weight'):
        #             # Min-max缩放到[0.1, 1.0]
        #             G[u][v]['weight'] = 5 + 10 * (w - min_weight) / (max_weight - min_weight)
        
        # 添加缺失的边，赋予较小的正权重（明显低于数据驱动的边）
        for node1 in nodes:
            for node2 in nodes:
                if not G.has_edge(node1, node2):
                    G.add_edge(node1, node2, weight=1)  # 非常小的正权重
        
        Gs.append(G)
        
    return Gs


def read_meta_datasets_i(window,root_path,data_path):
    labels = pd.read_csv(os.path.join(root_path,data_path))
    # print(group_idx)
    labels = labels.set_index("name")
    sdate = date(2020, 2, 24)
    edate = date(2020, 4, 24)
    #--- series of graphs and their respective dates
    delta = edate - sdate
    dates = [sdate + timedelta(days=i) for i in range(delta.days+1)]
    dates = [str(date) for date in dates]
    labels = labels.loc[:,dates]    #labels.sum(1).values>10   
    Gs = generate_graphs_tmp(dates,"IT")
    gs_adj = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs]

    labels = labels.loc[list(Gs[0].nodes()),:]

    features = generate_new_features(Gs ,labels ,dates ,window)
    y = list()
    for i,G in enumerate(Gs):
        y.append(list())
        for node in G.nodes():
            y[i].append(labels.loc[node,dates[i]])
    return labels, gs_adj, features, y

def read_meta_datasets_e(window,root_path,data_path):
    labels = pd.read_csv(os.path.join(root_path,data_path))
    # print(group_idx)
    labels = labels.set_index("name")
    sdate = date(2020, 3, 13)
    edate = date(2020, 5, 12)
    #--- series of graphs and their respective dates
    delta = edate - sdate
    dates = [sdate + timedelta(days=i) for i in range(delta.days+1)]
    dates = [str(date) for date in dates]
    labels = labels.loc[:,dates]    #labels.sum(1).values>10   
    Gs = generate_graphs_tmp(dates,"EN")
    gs_adj = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs]

    labels = labels.loc[list(Gs[0].nodes()),:]

    features = generate_new_features(Gs ,labels ,dates ,window)
    y = list()
    for i,G in enumerate(Gs):
        y.append(list())
        for node in G.nodes():
            y[i].append(labels.loc[node,dates[i]])
    return labels, gs_adj, features, y

def read_meta_datasets_f(window,root_path,data_path):
    labels = pd.read_csv(os.path.join(root_path,data_path))
    # print(group_idx)
    labels = labels.set_index("name")
    sdate = date(2020, 3, 10)
    edate = date(2020, 5, 12)
    #--- series of graphs and their respective dates
    delta = edate - sdate
    dates = [sdate + timedelta(days=i) for i in range(delta.days+1)]
    dates = [str(date) for date in dates]
    labels = labels.loc[:,dates]    #labels.sum(1).values>10   
    Gs = generate_graphs_tmp(dates,"FR")
    gs_adj = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs]

    labels = labels.loc[list(Gs[0].nodes()),:]

    features = generate_new_features(Gs ,labels ,dates ,window)
    y = list()
    for i,G in enumerate(Gs):
        y.append(list())
        for node in G.nodes():
            y[i].append(labels.loc[node,dates[i]])
    return labels, gs_adj, features, y

def read_meta_datasets_es(window,root_path,data_path):
    labels = pd.read_csv(os.path.join(root_path,data_path))
    # print(group_idx)
    labels = labels.set_index("name")
    sdate = date(2020, 3, 12)
    edate = date(2020, 5, 12)
    #--- series of graphs and their respective dates
    delta = edate - sdate
    dates = [sdate + timedelta(days=i) for i in range(delta.days+1)]
    dates = [str(date) for date in dates]
    labels = labels.loc[:,dates]    #labels.sum(1).values>10   
    Gs = generate_graphs_tmp(dates,"ES")
    gs_adj = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs]

    labels = labels.loc[list(Gs[0].nodes()),:]

    features = generate_new_features(Gs ,labels ,dates ,window)
    y = list()
    for i,G in enumerate(Gs):
        y.append(list())
        for node in G.nodes():
            y[i].append(labels.loc[node,dates[i]])
    return labels, gs_adj, features, y
class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class Dataset_EpiGNN(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='italy_labels.csv', scale=False):
        """
        时空序列数据集类
        
        参数:
            meta_graphs: np.ndarray 邻接矩阵序列 [总天数, 节点数, 节点数] (185, 20, 20)
            meta_features: np.ndarray 节点特征序列 [总天数, 节点数, 窗口长度] (185, 20, 3)
            meta_y: np.ndarray 标签序列 [总天数, 节点数] (185, 20)
            flag: str 数据集类型 ('train', 'val', 'test')
            seq_len: int 输入序列长度（时间步数）
            pred_len: int 预测序列长度
            window: int 特征窗口大小
        """
        self.seq_len = size[0]      # 输入序列长度 18
        self.pred_len = size[1]    # 预测长度 15
        self.window = size[2]        # 特征窗口大小 3   
        if flag == 'test':
            self.token_len = size[3]
        self.root_path = root_path
        self.data_path = data_path
       
        # 数据校验
        if self.data_path == "italy_labels.csv":
            if flag != 'test':
                meta_labs, meta_graphs, meta_features, meta_y = read_meta_datasets_i(self.seq_len,self.root_path,self.data_path)
            else:
                meta_labs, meta_graphs, meta_features, meta_y = read_meta_datasets_i(self.seq_len,self.root_path,self.data_path)
        elif self.data_path == "england_labels.csv":
            if flag != 'test':
                meta_labs, meta_graphs, meta_features, meta_y = read_meta_datasets_e(self.seq_len,self.root_path,self.data_path)
            else:
                meta_labs, meta_graphs, meta_features, meta_y = read_meta_datasets_e(self.seq_len,self.root_path,self.data_path)
        elif self.data_path == "france_labels.csv":
            if flag != 'test':
                meta_labs, meta_graphs, meta_features, meta_y = read_meta_datasets_f(self.seq_len,self.root_path,self.data_path)
            else:   
                meta_labs, meta_graphs, meta_features, meta_y = read_meta_datasets_f(self.seq_len,self.root_path,self.data_path)
        elif self.data_path =='spain_labels.csv':
            if flag != 'test':
                meta_labs, meta_graphs, meta_features, meta_y = read_meta_datasets_es(self.seq_len,self.root_path,self.data_path)
            else:
                meta_labs, meta_graphs, meta_features, meta_y = read_meta_datasets_es(self.seq_len,self.root_path,self.data_path)
        meta_graphs = torch.tensor(meta_graphs)
        meta_graphs[:] = meta_graphs[:] / meta_graphs[:].max()  # 归一化到[0, 1]
        meta_features = torch.tensor(meta_features)
        meta_y = torch.tensor(meta_y)
        # y_transposed = meta_y.t()
        if not meta_graphs.is_floating_point():
            meta_graphs = meta_graphs.float()
        # 添加微小常数防止数值不稳定
        eps = 1e-10
        # 计算度矩阵的根号倒数
        deg = torch.sum(meta_graphs, dim=2)  # 节点出度 [b, n]
        deg_inv_sqrt = torch.where(deg > 0, 1.0 / torch.sqrt(deg + eps), torch.zeros_like(deg))

        meta_graphs = deg_inv_sqrt.unsqueeze(-1) * meta_graphs * deg_inv_sqrt.unsqueeze(1)
        if not meta_y.is_floating_point():
            meta_y = meta_y.float()
        self.mn = meta_y.mean(dim = 0)
        self.scaler = StandardScaler(mean=meta_y.mean(), std=meta_y.std())
        self.num_nodes = meta_y.shape[1]       # 节点数量
        # print('self.num_nodes',self.num_nodes)
        self.flag = flag            # 数据集类型
        # 按时间划分数据集
        total_days = meta_features.shape[0] 
        # print("total_days",total_days)
        train_days = int((total_days-self.seq_len) *0.6 -self.seq_len)
        # 计算各数据集边界
        train_end = train_days + 2*self.seq_len
        val_end = int((total_days-self.seq_len) *0.8+self.seq_len)

        
        # 边界索引映射
        border_map = {
            'train': (0, train_end),
            'val': (train_end - self.seq_len, val_end),
            'test': (val_end - self.seq_len, total_days+1)
        }
        
        # 获取当前数据集的边界
        self.start_idx, self.end_idx = border_map[flag]
        # print("self.start_idx",self.start_idx)
        # print("self.end_idx",self.end_idx)
        
        # 切片获取子数据集
        self.graphs = meta_graphs[self.start_idx:self.end_idx]       # 邻接矩阵序列

        # print("self.graphs.shape",self.graphs.shape)
        self.features = meta_features[self.start_idx:self.end_idx]  # 节点特征序列
        # print("self.features.shape",self.features.shape)
        self.labels = meta_y[self.start_idx:self.end_idx]           # 标签序列
        # print("self.labels.shape",self.labels[12])
    
        # 计算有效样本数 (考虑序列长度和预测长度)
        self.valid_length = len(self.features) - 2 * self.seq_len + 1

    def __len__(self):
        """返回数据集总样本数"""
        return self.valid_length

    def __getitem__(self, index):
        """
        获取单个样本
        
        返回:
            x_feat: 输入特征 [seq_len, 节点数, window]
            x_graph: 邻接矩阵 [seq_len, 节点数, 节点数] 
            y: 预测标签 [pred_len, 节点数]
        """
        # 计算时间窗口索引
        t_start = index
        t_end = t_start + self.seq_len
        pred_start = t_end
        pred_end = t_end + self.seq_len

        # 获取输入特征 (使用预先计算好的窗口特征)
        x_feat1 = self.features[t_start:t_end]  # [seq_len, 20, 3]
        x_feat2 = self.features[pred_start:pred_end]
        
        # 获取对应时间的邻接矩阵序列
        x_graph1 = self.graphs[t_start:t_end]   # [seq_len, 20, 20]
        x_graph2 = self.graphs[pred_start:pred_end]
        
        # 获取预测标签
        y = self.labels[pred_start:pred_end]   # [pred_len, 20]

        # 转换为Tensor
        x_feat = torch.tensor(x_feat1,dtype=torch.float32)
        x_feat2 = torch.tensor(x_feat2,dtype=torch.float32)
        x_graph = torch.tensor(x_graph1,dtype=torch.float32)
        x_graph2 = torch.tensor(x_graph2,dtype=torch.float32)
        y = torch.tensor(y,dtype=torch.float32)


        return x_feat, x_feat2, x_graph, x_graph2,y

    def get_full_graph(self):
        """获取完整时间段的图结构(用于图神经网络初始化)"""
        return torch.FloatTensor(self.graphs)

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv',
                 scale=True, seasonal_patterns=None, drop_short=False):
        self.seq_len = size[0] #672
        self.label_len = size[1] #96*6=576
        self.pred_len = size[2]  #96
        self.token_len = self.seq_len - self.label_len #96
        self.token_num = self.seq_len // self.token_len #7
        self.flag = flag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag] #0

        self.scale = scale #True

        self.root_path = root_path #./dataset/ETT-small/
        self.data_path = data_path #ETTh1.csv
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]  # 1
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1  

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]  
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]   #0
        border2 = border2s[self.set_type]   #8928 12 * 30 * 24

        cols_data = df_raw.columns[1:] #all
        df_data = df_raw[cols_data] #all

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        data_name = self.data_path.split('.')[0] #ETTh1
        self.data_stamp = torch.load(os.path.join(self.root_path, f'{data_name}.pt'))   
        self.data_stamp = self.data_stamp[border1:border2] 
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = self.data_stamp[s_begin:s_end:self.token_len]
        seq_y_mark = self.data_stamp[s_end:r_end:self.token_len]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv',
                 scale=True, seasonal_patterns=None, drop_short=False):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
            

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        data_name = self.data_path.split('.')[0]
        self.data_stamp = torch.load(os.path.join(self.root_path, f'{data_name}.pt'))
        self.data_stamp = self.data_stamp[border1:border2]
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = self.data_stamp[s_begin:s_end:self.token_len]
        seq_y_mark = self.data_stamp[s_end:r_end:self.token_len]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Solar(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='ETTh1.csv',
                 seasonal_patterns=None, scale=True, drop_short=False):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.enc_in = self.data_x.shape[-1]
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len
        
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id+1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id+1]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return (len(self.data_x) - self.seq_len - self.pred_len + 1) * self.enc_in

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, root_path, flag='pred', size=None, data_path='ETTh1.csv',
                 scale=False, inverse=False, seasonal_patterns='Yearly', drop_short=False):
        self.scale = scale
        self.inverse = inverse
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           cut_point - self.label_len:min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class Dataset_TSF(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path=None,
                 scale=True, seasonal_patterns=None, drop_short=False):
        
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.pred_len
        self.context_len = 4 * self.token_len
        print(self.seq_len, self.label_len, self.pred_len)
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.root_path = root_path
        self.data_path = data_path
        self.drop_short = drop_short
        self.timeseries = self.__read_data__()


    def __read_data__(self):
        df, _, _, _, _ = convert_tsf_to_dataframe(os.path.join(self.root_path, self.data_path))
        def dropna(x):
            return x[~np.isnan(x)]
        timeseries = [dropna(ts).astype(np.float32) for ts in df.series_value]
        if self.drop_short:
            timeseries = [ts for ts in timeseries if ts.shape[0] > self.context_len]
        self.tot_len = 0
        self.len_seq = []
        self.seq_id = []
        for i in range(len(timeseries)):
            res_len = max(self.pred_len + self.seq_len - timeseries[i].shape[0], 0)
            pad_zeros = np.zeros(res_len)
            timeseries[i] = np.hstack([pad_zeros, timeseries[i]])

            _len = timeseries[i].shape[0]
            train_len = _len-self.pred_len
            border1s = [0,                          train_len - self.seq_len - self.pred_len, train_len-self.seq_len]
            border2s = [train_len - self.pred_len,  train_len,                                _len]
            
            curr_len = border2s[self.set_type] - max(border1s[self.set_type], 0) - self.pred_len - self.seq_len + 1
            curr_len = max(0, curr_len)
            
            self.len_seq.append(np.zeros(curr_len) + self.tot_len)
            self.seq_id.append(np.zeros(curr_len) + i)
            self.tot_len += curr_len
            
        self.len_seq = np.hstack(self.len_seq)
        self.seq_id = np.hstack(self.seq_id)

        return timeseries

    def __getitem__(self, index):
        len_seq = self.len_seq[index]
        seq_id = int(self.seq_id[index])
        index = index - int(len_seq)

        _len = self.timeseries[seq_id].shape[0]
        train_len = _len - self.pred_len
        border1s = [0,                          train_len - self.seq_len - self.pred_len, train_len-self.seq_len]

        s_begin = index + border1s[self.set_type]
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = s_end + self.pred_len

        data_x = self.timeseries[seq_id][s_begin:s_end]
        data_y = self.timeseries[seq_id][r_begin:r_end]
        data_x = np.expand_dims(data_x, axis=-1)
        data_y = np.expand_dims(data_y, axis=-1)

        return data_x, data_y, data_x, data_y

    def __len__(self):
        return self.tot_len

class Dataset_TSF_ICL(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path=None,
                 scale=True, seasonal_patterns=None, drop_short=True):
        
        self.pred_len = size[2]
        self.token_len = self.pred_len
        self.context_len = 4 * self.token_len

        self.root_path = root_path
        self.data_path = data_path
        self.timeseries = self.__read_data__()

    def __read_data__(self):
        df, _, _, _, _ = convert_tsf_to_dataframe(os.path.join(self.root_path, self.data_path))
        def dropna(x):
            return x[~np.isnan(x)]
        timeseries = [dropna(ts).astype(np.float32) for ts in df.series_value]
        timeseries = [ts for ts in timeseries if ts.shape[0] > self.context_len]
        return timeseries

    # we uniformly adopting the first time points of the time series as the corresponding prompt.
    def __getitem__(self, index):        
        data_x1 = self.timeseries[index][:2*self.token_len]
        data_x2 = self.timeseries[index][-2*self.token_len:-1*self.token_len]
        data_x = np.concatenate((data_x1, data_x2))
        data_y = self.timeseries[index][-1*self.token_len:]
        data_x = np.expand_dims(data_x, axis=-1)
        data_y = np.expand_dims(data_y, axis=-1)
        return data_x, data_y, data_x, data_y

    def __len__(self):
        return len(self.timeseries)

class Dataset_Preprocess(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 data_path='ETTh1.csv', scale=True, seasonal_patterns=None):
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.token_len = self.seq_len - self.label_len
        self.token_num = self.seq_len // self.token_len
        self.flag = flag
        self.data_set_type = data_path.split('.')[0]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.tot_len = len(self.data_stamp)

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        df_stamp = df_raw[['date']]
        df_stamp['date'] = pd.to_datetime(df_stamp.date).apply(str)
        self.data_stamp = df_stamp['date'].values
        self.data_stamp = [str(x) for x in self.data_stamp]
        

    def __getitem__(self, index):
        s_begin = index % self.tot_len
        s_end = s_begin + self.token_len
        start = datetime.datetime.strptime(self.data_stamp[s_begin], "%Y-%m-%d %H:%M:%S")
        if self.data_set_type in ['traffic', 'electricity', 'ETTh1', 'ETTh2']:
            end = (start + datetime.timedelta(hours=self.token_len-1)).strftime("%Y-%m-%d %H:%M:%S")
        elif self.data_set_type == 'weather':
            end = (start + datetime.timedelta(minutes=10*(self.token_len-1))).strftime("%Y-%m-%d %H:%M:%S")
        elif self.data_set_type in ['ETTm1', 'ETTm2']:
            end = (start + datetime.timedelta(minutes=15*(self.token_len-1))).strftime("%Y-%m-%d %H:%M:%S")
        seq_x_mark = f"This is Time Series from {self.data_stamp[s_begin]} to {end}"
        return seq_x_mark

    def __len__(self):
        return len(self.data_stamp)
