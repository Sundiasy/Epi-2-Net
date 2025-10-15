import os
import torch
import datetime
import numpy as np
import pandas as pd
import networkx as nx
from datetime import date, timedelta
from torch.utils.data import Dataset
from data_provider.m4 import M4Dataset, M4Meta
# from sklearn.preprocessing import StandardScaler
from utils.tools import convert_tsf_to_dataframe

import warnings
warnings.filterwarnings('ignore')

def generate_new_features(Gs, labels, dates, window, scaled=False):
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
    for idx, G in enumerate(Gs):
        #  Features = population, coordinates, d past cases, one hot region
        H = np.zeros([G.number_of_nodes(),window]) 
        me = labs.loc[:, dates[:(idx)]].mean(1)
        sd = labs.loc[:, dates[:(idx)]].std(1) + 1

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
        print(nodes)
        exit()
        G.add_nodes_from(nodes)
        # 从CSV添加边及其原始权重
        edge_weights = []
        for row in d.iterrows():
            G.add_edge(row[1][0], row[1][1], weight=row[1][2])
            edge_weights.append(row[1][2])
        
        # 添加缺失的边，赋予较小的正权重（明显低于数据驱动的边）
        for node1 in nodes:
            for node2 in nodes:
                if not G.has_edge(node1, node2):
                    G.add_edge(node1, node2, weight=1)  # 非常小的正权重
        
        Gs.append(G)
        
    return Gs


def read_meta_datasets_i(window, root_path, data_path):   # load data of italy

    labels = pd.read_csv(os.path.join(root_path, data_path))
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

def read_meta_datasets_e(window, root_path, data_path):

    labels = pd.read_csv(os.path.join(root_path, data_path))
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

def read_meta_datasets_f(window, root_path, data_path):

    labels = pd.read_csv(os.path.join(root_path,data_path))
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

def read_meta_datasets_es(window, root_path, data_path):

    labels = pd.read_csv(os.path.join(root_path,data_path))
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
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class Dataset_Epi(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='italy_labels.csv', scale=False):
        """
            meta_graphs: np.ndarray 邻接矩阵序列 [总天数, 节点数, 节点数] 
            meta_features: np.ndarray 节点特征序列 [总天数, 节点数, 窗口长度] 
            meta_y: np.ndarray 标签序列 [总天数, 节点数] 
            flag: str 数据集类型 ('train', 'val', 'test')
            seq_len: int 输入序列长度（时间步数）
            pred_len: int 预测序列长度
            window: int 特征窗口大小
        """
        self.root_path = root_path
        self.data_path = data_path

        self.seq_len = size[0]       
        self.pred_len = size[1]      
        self.window = size[2]         
        if flag == 'test':
            self.token_len = size[3]
        
        if self.data_path == "italy_labels.csv":
            meta_labs, meta_graphs, meta_features, meta_y = read_meta_datasets_i(self.seq_len, self.root_path, self.data_path)
        elif self.data_path == "england_labels.csv":
            meta_labs, meta_graphs, meta_features, meta_y = read_meta_datasets_e(self.seq_len, self.root_path, self.data_path)
        elif self.data_path == "france_labels.csv":   
            meta_labs, meta_graphs, meta_features, meta_y = read_meta_datasets_f(self.seq_len, self.root_path, self.data_path)
        elif self.data_path =='spain_labels.csv':
            meta_labs, meta_graphs, meta_features, meta_y = read_meta_datasets_es(self.seq_len, self.root_path, self.data_path)
        
        meta_graphs = torch.tensor(meta_graphs)                 # shape [num_days, num_regions, num_regions]
        meta_graphs[:] = meta_graphs[:] / meta_graphs[:].max()  # normalization to [0,1]
        meta_features = torch.tensor(meta_features)             # shape [num_days, num_regions, seq_len]
        meta_y = torch.tensor(meta_y)                           # shape [num_days, num_regions]

        if not meta_graphs.is_floating_point():
            meta_graphs = meta_graphs.float()

        # 添加微小常数防止数值不稳定
        eps = 1e-10
        deg = torch.sum(meta_graphs, dim=2)  
        deg_inv_sqrt = torch.where(deg > 0, 1.0 / torch.sqrt(deg + eps), torch.zeros_like(deg))
        meta_graphs = deg_inv_sqrt.unsqueeze(-1) * meta_graphs * deg_inv_sqrt.unsqueeze(1)

        if not meta_y.is_floating_point():
            meta_y = meta_y.float()

        self.scaler = StandardScaler(mean=meta_y.mean(), std=meta_y.std())
        self.num_nodes = meta_y.shape[1]       
        self.flag = flag            

        total_days = meta_features.shape[0]
        train_days = int((total_days - 4 * self.seq_len) * 0.8)
        train_end = train_days + self.seq_len * 2
        val_end = train_end + self.seq_len + int(train_days / 8)
        
        # 边界索引映射
        border_map = {
            'train': (0, train_end),
            'val': (train_end - self.seq_len, val_end),
            'test': (val_end - self.seq_len, total_days + 1)
        }
        self.start_idx, self.end_idx = border_map[flag]
        
        # 切片获取子数据集
        self.graphs = meta_graphs[self.start_idx: self.end_idx]      
        self.features = meta_features[self.start_idx: self.end_idx]  
        self.labels = meta_y[self.start_idx: self.end_idx]           
        self.valid_length = len(self.features) - 2 * self.seq_len + 1

    def __len__(self):
        """返回数据集总样本数"""
        return self.valid_length

    def __getitem__(self, index):
        """
            x_feat: [seq_len, num_nodes, window]
            x_graph: [seq_len, num_nodes, num_nodes] 
            y: [pred_len, num_nodes]
        """
        t_start = index
        t_end = t_start + self.seq_len
        pred_start = t_end
        pred_end = t_end + self.seq_len

        x_feat1 = self.features[t_start: t_end]  
        x_feat2 = self.features[pred_start: pred_end]

        x_graph1 = self.graphs[t_start: t_end]   
        x_graph2 = self.graphs[pred_start: pred_end]
        
        y = self.labels[pred_start: pred_end]   

        x_feat = torch.tensor(x_feat1, dtype=torch.float32)
        x_feat2 = torch.tensor(x_feat2, dtype=torch.float32)
        x_graph = torch.tensor(x_graph1, dtype=torch.float32)
        x_graph2 = torch.tensor(x_graph2, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return x_feat, x_feat2, x_graph, x_graph2, y

    def get_full_graph(self):
        
        return torch.FloatTensor(self.graphs)

  
        






