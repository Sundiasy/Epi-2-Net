import time
import torch
import torch.nn as nn

from torch.nn.modules.rnn import GRU
from ode_func import ODEFunc
from diffeq_solver import DiffeqSolver
from utills import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class EncoderAttrs:
    
    def __init__(self, adj_mx, **model_kwargs):
        self.adj_mx = adj_mx                        
        self.num_nodes = adj_mx.shape[0]          
        self.num_edges = (adj_mx > 0.).sum()          
        self.gcn_step = int(model_kwargs.get('gcn_step', 2))       
        self.filter_type = model_kwargs.get('filter_type')
        self.num_rnn_layers = int(model_kwargs.get('num_rnn_layers', 1))
        self.rnn_units = int(model_kwargs.get('rnn_units'))
        self.latent_dim = int(model_kwargs.get('latent_dim', 4))

class Epi2NetModel(nn.Module, EncoderAttrs):
    
    def __init__(self, adj_mx,logger,args, **model_kwargs):
        nn.Module.__init__(self)
        EncoderAttrs.__init__(self, adj_mx, **model_kwargs)
        self._logger = logger
        self.args = args
        self.batch_size = int(args.batch_size)
        # RNN Encoder
        self.encoder_z0 = Encoder_z0_RNN(adj_mx, args,**model_kwargs)
        # self.adj_mx =  adj_mx
        # ODE solver
        self.n_traj_samples = int(model_kwargs.get('n_traj_samples', 1))
        self.ode_method = model_kwargs.get('ode_method', 'euler')
        self.atol = float(model_kwargs.get('odeint_atol', 1e-4))
        self.rtol = float(model_kwargs.get('odeint_rtol', 1e-3))
        self.num_gen_layer = int(model_kwargs.get('gen_layers', 1))
        self.ode_gen_dim = int(model_kwargs.get('gen_dim', 64))
        ode_set_str = "ODE setting --latent {} --samples {} --method {} \
            --atol {:6f} --rtol {:6f} --gen_layer {} --gen_dim {}".format(\
                self.latent_dim, self.n_traj_samples, self.ode_method, \
                self.atol, self.rtol, self.num_gen_layer, self.ode_gen_dim)
        self.odefunc = ODEFunc(self.ode_gen_dim, self.latent_dim, self.adj_mx,
                          self.gcn_step, self.num_nodes,self.batch_size, filter_type=self.filter_type).to(device)
        # Dataset
        

        self._logger.info(ode_set_str)
        self.save_latent = bool(model_kwargs.get('save_latent', False))
        self.latent_feat = None # used to extract the latent feature

        # Decoder
        self.horizon = int(model_kwargs.get('horizon', 1))
        self.out_feat = int(model_kwargs.get('output_dim', 1))
        self.decoder = Decoder(self.out_feat, adj_mx, self.num_nodes, self.num_edges,).to(device)
        # self.decoder = DecoderMLP(self.out_feat, self.latent_dim, self.num_nodes).to(device)

    def forward(self, inputs, labels=None, adj_x=None, batches_seen=0):
        """
        seq2seq forward pass
        :param inputs: shape (seq_len, batch_size, num_nodes * input_dim)
        :param labels: shape (horizon, batch_size, num_nodes * output_dim)
        :param batches_seen: batches seen till now
        :return: outputs: (self.horizon, batch_size, self.num_edges * self.output_dim)
        """
        perf_time = time.time()
        # shape: first_point_mu  [1, batch, num_nodes * latent_dim] latent_dim 4
        #        first_point_std [1, batch, num_nodes * latent_dim]
        #        last_wind_vars  [batch, num_nodes, wind_dim] wind_dim 2
        first_point_mu, first_point_std = self.encoder_z0(inputs) 
        # self._logger.debug("Recognition complete with {:.1f}s".format(time.time() - perf_time))

        # sample 'n_traj_samples' trajectory
        perf_time = time.time()
        means_z0 = first_point_mu.repeat(self.n_traj_samples, 1, 1)
        sigma_z0 = first_point_std.repeat(self.n_traj_samples, 1, 1)
        first_point_enc = sample_standard_gaussian(means_z0, sigma_z0) # [3, batch, num_nodes * latent_dim]
         
        time_steps_to_predict = torch.arange(start=0, end=self.horizon, step=1).float().to(device)
        time_steps_to_predict = time_steps_to_predict / len(time_steps_to_predict)
    
        # start = time.time()
        self.diffeq_solver = DiffeqSolver(self.odefunc, self.ode_method, self.latent_dim, odeint_rtol=self.rtol,
                                          odeint_atol=self.atol)
        # print('Time = ',str(time.time()-start))
        # Shape of sol_ys (horizon, n_traj_samples, batch_size, self.num_nodes * self.latent_dim)
        sol_ys, fe = self.diffeq_solver(first_point_enc, time_steps_to_predict,adj_x)
        self._logger.debug("ODE solver complete with {:.1f}s".format(time.time() - perf_time))
        
        if(self.save_latent):
            # Shape of latent_feat (horizon, batch_size, self.num_nodes * self.latent_dim)
            self.latent_feat = torch.mean(sol_ys.detach(), axis=1)

        perf_time = time.time()
        outputs = self.decoder(sol_ys)
        self._logger.debug("Decoder complete with {:.1f}s".format(time.time() - perf_time))

        # if batches_seen == 0:
        #     self._logger.info(
        #         "Total trainable parameters {}".format(count_parameters(self))
        #     )
        return outputs, fe

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 辅助模块 ---

class Chomp1d(nn.Module):
    """
    一个简单的模块，用于裁剪掉卷积后多余的 padding，实现因果卷积。
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TCNBlock(nn.Module):
    """
    TCN 的基本残差块。
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Encoder_z0_RNN(nn.Module, EncoderAttrs):
    
    def __init__(self, adj_mx, args, **model_kwargs):
        nn.Module.__init__(self)
        EncoderAttrs.__init__(self, adj_mx, **model_kwargs)
        self.recg_type = model_kwargs.get('recg_type', 'gru') # gru

        if(self.recg_type == 'gru'): # gru settings
            self.input_dim = int(model_kwargs.get('input_dim', 1))
            self.gru_rnn = GRU(self.input_dim, self.rnn_units).to(device)
        else:
            raise NotImplementedError("The recognition net only support 'gru'.")

        # hidden to z0 settings
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(self.rnn_units, 50),
            nn.Tanh(),
            nn.Linear(50, self.latent_dim * 2),)

        init_network_weights(self.hiddens_to_z0)

    def forward(self, inputs):
        """
        encoder forward pass on t time steps
        :param inputs: shape (seq_len, batch_size, num_edges * input_var)
        :return: mean, std: # shape (n_samples=1, batch_size, self.latent_dim)
        """
        if(self.recg_type == 'gru'):# shape of outputs: (seq_len, batch, num_senor * rnn_units)
            seq_len, batch_size = inputs.size(0), inputs.size(1)
            inputs = inputs.unsqueeze(-1)  # [21,1, 105, 1]
            seq_len,batch,num_nodes,_ = inputs.shape
            inputs = inputs.reshape(-1, inputs.shape[1] * inputs.shape[2], 1) # [24, 1120, 6]

            pm25 = inputs[:,:,0].unsqueeze(-1)   
            # wind_vars = inputs[:,:,-2:]      
            outputs, _ = self.gru_rnn(pm25) 
             
            last_output = outputs[-1]     
            last_output = torch.reshape(last_output, (batch, num_nodes, -1))  # (batch_size, num_nodes, rnn_units) 
            # last_wind_vars = torch.reshape(wind_vars[-1], (batch_size, self.num_nodes, -1)) #(batch_size, num_nodes, wind_dim) 
        else:
            raise NotImplementedError("The recognition net only support 'gru'.")

        mean, std = split_last_dim(self.hiddens_to_z0(last_output))
        mean = mean.reshape(batch_size, -1) # (batch_size, num_nodes * latent_dim)--(32, 140)
        std = std.reshape(batch_size, -1)   # (batch_size, num_nodes * latent_dim)--(32, 140)
        std = std.abs()

        assert(not torch.isnan(mean).any())
        assert(not torch.isnan(std).any())

        return mean.unsqueeze(0), std.unsqueeze(0)


class Encoder_z0_Transformer(nn.Module, EncoderAttrs):

    def __init__(self, adj_mx, args, **model_kwargs):
        nn.Module.__init__(self)
        EncoderAttrs.__init__(self, adj_mx, **model_kwargs)
        self.input_dim = int(model_kwargs.get('input_dim', 1))
        self.d_model = self.rnn_units 
        nhead = model_kwargs.get('nhead', 8)
        num_layers = model_kwargs.get('num_encoder_layers', 3)

        self.input_proj = nn.Linear(self.input_dim, self.d_model)


        self.pos_encoder = PositionalEncoding(self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(self.d_model, 50),
            nn.Tanh(),
            nn.Linear(50, self.latent_dim * 2),
        )
        init_network_weights(self.input_proj)
        init_network_weights(self.hiddens_to_z0)

    def forward(self, inputs):
        """
        encoder forward pass on t time steps
        :param inputs: shape (seq_len, batch_size, num_nodes * input_dim)
        :return: mean, std: # shape (n_samples=1, batch_size, num_nodes * latent_dim)
        """
        seq_len, batch_size, _ = inputs.shape
        # [seq_len, batch, num_nodes * input_dim] -> [seq_len, batch * num_nodes, input_dim]
        inputs = inputs.reshape(seq_len, batch_size * self.num_nodes, self.input_dim)

        # 1. 通过输入投射层
        # [seq_len, batch * num_nodes, input_dim] -> [seq_len, batch * num_nodes, d_model]
        inputs_proj = self.input_proj(inputs)

        # 2. 添加位置编码
        inputs_pos = self.pos_encoder(inputs_proj)

        # 3. 通过 Transformer Encoder
        # TransformerEncoder 的输入格式就是 (S, N, E)，即 (seq_len, batch, embedding_dim)
        # output shape: (seq_len, batch * num_nodes, d_model)
        transformer_output = self.transformer_encoder(inputs_pos)
        
        # 取最后一个时间步的输出作为序列的表征
        last_output = transformer_output[-1] # shape: (batch * num_nodes, d_model)
        
        # [batch * num_nodes, d_model] -> [batch, num_nodes, d_model]
        last_output = torch.reshape(last_output, (batch_size, self.num_nodes, -1))
        
        # 4. 将表征映射到隐空间的均值和标准差
        mean, std = split_last_dim(self.hiddens_to_z0(last_output))
        
        # [batch, num_nodes, latent_dim] -> [batch, num_nodes * latent_dim]
        mean = mean.reshape(batch_size, -1)
        std = std.reshape(batch_size, -1)
        std = std.abs()

        assert not torch.isnan(mean).any()
        assert not torch.isnan(std).any()

        # 添加 sample 维度以匹配输出要求
        return mean.unsqueeze(0), std.unsqueeze(0)
    
class Encoder_z0_Avg(nn.Module, EncoderAttrs):

    def __init__(self, adj_mx, args, **model_kwargs):
        nn.Module.__init__(self)
        EncoderAttrs.__init__(self, adj_mx, **model_kwargs)
        self.input_dim = int(model_kwargs.get('input_dim', 1))
        # 在这个模型中，没有时序建模主干网络
        # hiddens_to_z0 的输入维度是原始特征的维度 self.input_dim
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(self.input_dim, 50), # 输入维度变为 self.input_dim
            nn.Tanh(),
            nn.Linear(50, self.latent_dim * 2),
        )
        init_network_weights(self.hiddens_to_z0)

    def forward(self, inputs):
        """
        encoder forward pass on t time steps
        :param inputs: shape (seq_len, batch_size, num_nodes * input_dim)
        :return: mean, std: # shape (n_samples=1, batch_size, num_nodes * latent_dim)
        """
        seq_len, batch_size, _ = inputs.shape
        # [seq_len, batch, num_nodes * input_dim] -> [seq_len, batch * num_nodes, input_dim]
        inputs = inputs.reshape(seq_len, batch_size * self.num_nodes, self.input_dim)
        
        # 在时间维度 (dim=0) 上求平均值，得到每个序列的表征
        # [seq_len, batch * num_nodes, input_dim] -> [batch * num_nodes, input_dim]
        avg_output = torch.mean(inputs, dim=0)

        # [batch * num_nodes, input_dim] -> [batch, num_nodes, input_dim]
        avg_output = torch.reshape(avg_output, (batch_size, self.num_nodes, -1))
        
        # 将平均表征映射到隐空间的均值和标准差
        mean, std = split_last_dim(self.hiddens_to_z0(avg_output))
        
        # [batch, num_nodes, latent_dim] -> [batch, num_nodes * latent_dim]
        mean = mean.reshape(batch_size, -1)
        std = std.reshape(batch_size, -1)
        std = std.abs()

        assert not torch.isnan(mean).any()
        assert not torch.isnan(std).any()

        # 添加 sample 维度以匹配输出要求
        return mean.unsqueeze(0), std.unsqueeze(0)

class Encoder_z0_TCN(nn.Module, EncoderAttrs):

    def __init__(self, adj_mx, args, **model_kwargs):
        nn.Module.__init__(self)
        EncoderAttrs.__init__(self, adj_mx, **model_kwargs)
        
        self.input_dim = int(model_kwargs.get('input_dim', 1))
        
        # TCN 主干网络
        tcn_channels = model_kwargs.get('tcn_channels', [32, self.rnn_units])
        kernel_size = model_kwargs.get('kernel_size', 7)
        
        layers = []
        num_levels = len(tcn_channels)
        # 动态构建 TCN block
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = self.input_dim if i == 0 else tcn_channels[i-1]
            out_channels = tcn_channels[i]
            layers += [TCNBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                padding=(kernel_size-1) * dilation_size)]
        
        self.tcn_net = nn.Sequential(*layers)

        # hidden to z0 的映射层，输入维度为 TCN 的最终输出通道数
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(tcn_channels[-1], 50),
            nn.Tanh(),
            nn.Linear(50, self.latent_dim * 2),
        )
        init_network_weights(self.hiddens_to_z0)

    def forward(self, inputs):
        """
        encoder forward pass on t time steps
        :param inputs: shape (seq_len, batch_size, num_nodes * input_dim)
        :return: mean, std: # shape (n_samples=1, batch_size, num_nodes * latent_dim)
        """
        seq_len, batch_size, _ = inputs.shape
        # [seq_len, batch, num_nodes * input_dim] -> [seq_len, batch * num_nodes, input_dim]
        inputs = inputs.reshape(seq_len, batch_size * self.num_nodes, self.input_dim)

        # TCN 需要 (N, C, L) 格式，当前是 (L, N, C)
        # N = batch * num_nodes, C = input_dim, L = seq_len
        inputs_permuted = inputs.permute(1, 2, 0) # -> (batch * num_nodes, input_dim, seq_len)
        
        # 通过 TCN 网络
        tcn_output = self.tcn_net(inputs_permuted) # -> (batch * num_nodes, tcn_channels[-1], seq_len)
        
        # 取最后一个时间步的输出作为序列的表征
        last_output = tcn_output[:, :, -1] # -> (batch * num_nodes, tcn_channels[-1])
        
        # [batch * num_nodes, rnn_units] -> [batch, num_nodes, rnn_units]
        last_output = torch.reshape(last_output, (batch_size, self.num_nodes, -1))
        
        # 将表征映射到隐空间的均值和标准差
        mean, std = split_last_dim(self.hiddens_to_z0(last_output))
        
        # [batch, num_nodes, latent_dim] -> [batch, num_nodes * latent_dim]
        mean = mean.reshape(batch_size, -1)
        std = std.reshape(batch_size, -1)
        std = std.abs()

        assert not torch.isnan(mean).any()
        assert not torch.isnan(std).any()

        # 添加 sample 维度以匹配输出要求
        return mean.unsqueeze(0), std.unsqueeze(0)

class Encoder_z0_LSTM(nn.Module, EncoderAttrs):
    
    def __init__(self, adj_mx, args, **model_kwargs):
        nn.Module.__init__(self)
        EncoderAttrs.__init__(self, adj_mx, **model_kwargs)
        
        # LSTM 主干网络
        self.input_dim = int(model_kwargs.get('input_dim', 1))
        self.lstm_rnn = nn.LSTM(self.input_dim, self.rnn_units)

        # hidden to z0 的映射层保持不变
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(self.rnn_units, 50),
            nn.Tanh(),
            nn.Linear(50, self.latent_dim * 2),
        )
        init_network_weights(self.hiddens_to_z0)

    def forward(self, inputs):
        """
        encoder forward pass on t time steps
        :param inputs: shape (seq_len, batch_size, num_nodes * input_dim)
        :return: mean, std: # shape (n_samples=1, batch_size, num_nodes * latent_dim)
        """
        seq_len, batch_size, _ = inputs.shape
        # [seq_len, batch, num_nodes * input_dim] -> [seq_len, batch * num_nodes, input_dim]
        inputs = inputs.reshape(seq_len, batch_size * self.num_nodes, self.input_dim)

        # LSTM 的输入和 GRU 类似
        # outputs shape: (seq_len, batch * num_nodes, rnn_units)
        # LSTM 返回 (output, (hidden_state, cell_state))
        outputs, _ = self.lstm_rnn(inputs)
        
        # 取最后一个时间步的输出作为序列的表征
        last_output = outputs[-1]  # shape: (batch * num_nodes, rnn_units)
        
        # [batch * num_nodes, rnn_units] -> [batch, num_nodes, rnn_units]
        last_output = torch.reshape(last_output, (batch_size, self.num_nodes, -1))
        
        # 将表征映射到隐空间的均值和标准差
        mean, std = split_last_dim(self.hiddens_to_z0(last_output))
        
        # [batch, num_nodes, latent_dim] -> [batch, num_nodes * latent_dim]
        mean = mean.reshape(batch_size, -1)
        std = std.reshape(batch_size, -1)
        std = std.abs()

        assert not torch.isnan(mean).any()
        assert not torch.isnan(std).any()

        # 添加 sample 维度以匹配输出要求
        return mean.unsqueeze(0), std.unsqueeze(0)
import torch
import torch.nn as nn

import torch
import torch.nn as nn

class DecoderMLP(nn.Module):
    def __init__(self, output_dim, latent_dim, num_nodes, hidden_dim=128, **kwargs):
        """
        :param output_dim: The dimension of the output for each node.
        :param latent_dim: The dimension of the latent representation for each node.
        :param num_nodes: The number of nodes in the graph.
        :param hidden_dim: The dimension of the hidden layer in the MLP.
        :param kwargs: Additional arguments (e.g., adj_mx, num_edges) for compatibility.
        """
        super(DecoderMLP, self).__init__()

        self.num_nodes = num_nodes
        self.output_dim = output_dim
        self.latent_dim = latent_dim

        # An MLP to project from latent_dim to output_dim with a hidden layer
        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim)
        )

    def forward(self, inputs):
        """
        :param inputs: (horizon, n_traj_samples, batch_size, num_nodes * latent_dim)
        :return outputs: (horizon, batch_size, num_nodes * output_dim), a result averaged over n_traj_samples.
        """
        assert(len(inputs.size()) == 4)
        horizon, n_traj_samples, batch_size, _ = inputs.size()

        # Reshape to isolate node features: (H, S, B, N, L)
        # H=horizon, S=n_traj_samples, B=batch_size, N=num_nodes, L=latent_dim
        inputs = inputs.view(horizon, n_traj_samples, batch_size, self.num_nodes, self.latent_dim)

        # Apply the MLP to the last dimension (the latent_dim)
        # Input: (H, S, B, N, L) -> Output: (H, S, B, N, O) where O=output_dim
        outputs = self.mlp(inputs)

        # Average over the trajectory samples (dimension 1)
        # Input: (H, S, B, N, O) -> Output: (H, B, N, O)
        outputs = torch.mean(outputs, dim=1)

        # Reshape to the required final output format
        # Input: (H, B, N, O) -> Output: (H, B, N * O)
        outputs = outputs.view(horizon, batch_size, -1)

        return outputs


class DecoderLinear(nn.Module):
    def __init__(self, output_dim, latent_dim, num_nodes, **kwargs):
        """
        :param output_dim: The dimension of the output for each node.
        :param latent_dim: The dimension of the latent representation for each node.
        :param num_nodes: The number of nodes in the graph.
        :param kwargs: Additional arguments (e.g., adj_mx, num_edges) for compatibility.
        """
        super(DecoderLinear, self).__init__()

        self.num_nodes = num_nodes
        self.output_dim = output_dim
        self.latent_dim = latent_dim

        # A single linear layer to project from latent_dim to output_dim
        self.projection_layer = nn.Linear(self.latent_dim, self.output_dim)

    def forward(self, inputs):
        """
        :param inputs: (horizon, n_traj_samples, batch_size, num_nodes * latent_dim)
        :return outputs: (horizon, batch_size, num_nodes * output_dim), a result averaged over n_traj_samples.
        """
        assert(len(inputs.size()) == 4)
        horizon, n_traj_samples, batch_size, _ = inputs.size()

        # Reshape to isolate node features: (H, S, B, N, L)
        # H=horizon, S=n_traj_samples, B=batch_size, N=num_nodes, L=latent_dim
        inputs = inputs.view(horizon, n_traj_samples, batch_size, self.num_nodes, self.latent_dim)

        # Apply the linear projection to the last dimension (the latent_dim)
        # Input: (H, S, B, N, L) -> Output: (H, S, B, N, O) where O=output_dim
        outputs = self.projection_layer(inputs)

        # Average over the trajectory samples (dimension 1)
        # Input: (H, S, B, N, O) -> Output: (H, B, N, O)
        outputs = torch.mean(outputs, dim=1)

        # Reshape to the required final output format
        # Input: (H, B, N, O) -> Output: (H, B, N * O)
        outputs = outputs.view(horizon, batch_size, -1)

        return outputs


class Decoder(nn.Module):
    def __init__(self, output_dim, adj_mx, num_nodes, num_edges):
        super(Decoder, self).__init__()

        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.output_dim = output_dim

    def forward(self, inputs):
        """
        :param inputs: (horizon, n_traj_samples, batch_size, num_nodes * latent_dim)
        :return outputs: (horizon, batch_size, num_nodes * output_dim), average result of n_traj_samples.
        """
        assert(len(inputs.size()) == 4)
        horizon, n_traj_samples, batch_size = inputs.size()[:3]
        inputs = inputs.reshape(horizon, n_traj_samples, batch_size, self.num_nodes, -1).transpose(-2, -1)
        latent_dim = inputs.size(-2)
        outputs = inputs.reshape(horizon, n_traj_samples, batch_size, latent_dim, self.num_nodes, self.output_dim)
        outputs = torch.mean(torch.mean(outputs, axis=3), axis=1)
        outputs = outputs.reshape(horizon, batch_size, -1)
    
        return outputs