import os
import datetime
import numpy as np
import pandas as pd
import torch
import networkx as nx
import warnings
from datetime import date, timedelta
from torch.utils.data import Dataset

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
    
    for idx, G in enumerate(Gs):
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
            d = pd.read_csv("Italy/graphs/"+country+"_"+date+".csv", header=None)
        elif country == "EN":
            d = pd.read_csv("England/graphs/"+country+"_"+date+".csv", header=None)
        elif country == "FR":
            d = pd.read_csv("France/graphs/"+country+"_"+date+".csv", header=None)
        elif country == "ES":
            d = pd.read_csv("Spain/graphs/"+country+"_"+date+".csv", header=None)
        
        G = nx.DiGraph()
        nodes = set(d[0].unique()).union(set(d[1].unique()))
        nodes = sorted(nodes)
        G.add_nodes_from(nodes)

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

class Dataset_EpiGNN(Dataset):
    def __init__(self, root_path, flag='train', size=None, data_path='italy_labels.csv',
                 scale=False, seasonal_patterns=None, drop_short=False):
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
                meta_labs, meta_graphs, meta_features, meta_y = read_meta_datasets_i(self.window,self.root_path,self.data_path)
            else:
                meta_labs, meta_graphs, meta_features, meta_y = read_meta_datasets_i(self.token_len,self.root_path,self.data_path)
        elif self.data_path == "england_labels.csv":
            if flag != 'test':
                meta_labs, meta_graphs, meta_features, meta_y = read_meta_datasets_e(self.window,self.root_path,self.data_path)
            else:
                meta_labs, meta_graphs, meta_features, meta_y = read_meta_datasets_e(self.token_len,self.root_path,self.data_path)
        elif self.data_path == "france_labels.csv":
            if flag != 'test':
                meta_labs, meta_graphs, meta_features, meta_y = read_meta_datasets_f(self.window,self.root_path,self.data_path)
            else:   
                meta_labs, meta_graphs, meta_features, meta_y = read_meta_datasets_f(self.token_len,self.root_path,self.data_path)
        elif self.data_path =='spain_labels.csv':
            if flag != 'test':
                meta_labs, meta_graphs, meta_features, meta_y = read_meta_datasets_es(self.window,self.root_path,self.data_path)
            else:
                meta_labs, meta_graphs, meta_features, meta_y = read_meta_datasets_es(self.token_len,self.root_path,self.data_path)
        meta_graphs = torch.tensor(meta_graphs)
        meta_features = torch.tensor(meta_features)
        meta_y = torch.tensor(meta_y)

        
        self.num_nodes = meta_y.shape[1]         # 节点数量
        # print('self.num_nodes',self.num_nodes)
        self.flag = flag            # 数据集类型
        # 按时间划分数据集
        total_days = meta_features.shape[0] 
        # print("total_days",total_days)
        
        # 计算各数据集边界

        train_end = total_days-self.window*2
        val_end = total_days-self.window

        
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
        self.valid_length = len(self.features) - self.seq_len - self.window + 1

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
        pred_start = t_end - self.pred_len
        pred_end = pred_start + self.window + self.pred_len

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

        return x_feat, x_feat2, x_graph, x_graph2, y

    def get_full_graph(self):
        """获取完整时间段的图结构(用于图神经网络初始化)"""
        return torch.FloatTensor(self.graphs)

if __name__ == '__main__':
    size = [18, 15, 3]
    root_path = 'Italy/'
    data_path='italy_labels.csv'
    flag='train'
    x_feat, x_feat2, x_graph, x_graph2, y = Dataset_EpiGNN(root_path, flag, size, data_path)[0]
    print(x_feat.shape)
