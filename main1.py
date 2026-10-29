from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from utils import *
from epi2net_supervisor import Epi2NetSupervisor
import numpy as np
import torch
from data_provider.data_factory import get_adj
import yaml
import os
import random

def get_mean_std(data_list):
    return data_list.mean(), data_list.std()

def set_seed(seed):
    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  

def main(args):
    print("Generating Adjacency Matrix")
    seed = 2035
    set_seed(seed) 
    adj_mx = get_adj('England') # Massachusetts # California
    rmse_list, mae_list, mape_list, preds_list, truths_list = [], [], [], [], []
    for exp_idx in range(args.exp_repeat):
        print('\nNo.%2d experiment ~~~' % exp_idx)
        with open(args.config_filename) as f:
            supervisor_config = yaml.load(f, Loader=yaml.Loader)
            
            base_dir = supervisor_config['base_dir']
            supervisor_config['base_dir'] = os.path.join(base_dir, 'exp_'+str(exp_idx)) 
            base_model = supervisor_config['base_model_dir']
            supervisor_config['base_model_dir'] = os.path.join(base_model, 'exp_'+str(exp_idx)) 
            supervisor = Epi2NetSupervisor(adj_mx=adj_mx,args = args, **supervisor_config)
            supervisor.train()
            mae, mape, rmse = supervisor.evaluate_more('test')

            mae_list.append(mae)
            mape_list.append(mape)
            rmse_list.append(rmse)
            
            
    mae_list = np.array(mae_list) #(num_exp, num_seq)
    mape_list = np.array(mape_list)
    rmse_list = np.array(rmse_list)

    preds_list = np.array(preds_list)
    truths_list = np.array(truths_list)
    mean_preds = np.mean(preds_list, axis=0)
    mean_truths = np.mean(truths_list, axis=0)
    outputs = {'prediction': mean_preds, 'truth': mean_truths}
    np.savez_compressed(args.output_filename, **outputs)
    print('Mean predictions saved as {}.'.format(args.output_filename))
   
    seq_len = [5,10]
    print('--------- Epi2Net Final Results ------------' )
    for i, seq in enumerate(seq_len):
        print('Evaluation seq {}:'.format(seq))
        print('MAE | mean: {:.4f} std: {:.4f}'.format(get_mean_std(mae_list[:,i])[0], get_mean_std(mae_list[:,i])[1]))
        print('MAPE | mean: {:.4f} std: {:.4f}'.format(get_mean_std(mape_list[:,i])[0], get_mean_std(mape_list[:,i])[1]))
        print('RMSE | mean: {:.4f} std: {:.4f}\n'.format(get_mean_std(rmse_list[:,i])[0], get_mean_std(rmse_list[:,i])[1]))
    
    output_text = ''
    output_text += '--------- Epi2Net Final Results ------------\n'
    for i, seq in enumerate(seq_len):
        output_text += 'Evaluation seq {}:\n'.format(seq)
        output_text += 'MAE | mean: {:.4f} std: {:.4f}\n'.format(get_mean_std(mae_list[:,i])[0], get_mean_std(mae_list[:,i])[1])
        output_text += 'MAPE | mean: {:.4f} std: {:.4f}\n'.format(get_mean_std(mape_list[:,i])[0], get_mean_std(mape_list[:,i])[1])
        output_text += 'RMSE | mean: {:.4f} std: {:.4f}\n\n'.format(get_mean_std(rmse_list[:,i])[0], get_mean_std(rmse_list[:,i])[1])

    # Write the output text to a file
    with open('logs/Epi2Net_results.txt', 'w') as file:
        file.write(output_text)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor_filename', type=str, default='data/station.csv',help='Sensor Data.')
    parser.add_argument('--config_filename', default='config.yaml', type=str, help='Configuration filename for restoring the model.')
    parser.add_argument('--output_filename', default='models/epi2net_predictions.npz')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    parser.add_argument('--exp_repeat', type=int, default=1, help='Number of experiments.')
    parser.add_argument('--data', type=str,  default='epidataset', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/England/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='england_labels.csv', help='data file') # COVIDinCalifornia.json
    parser.add_argument('--node_num', type=int, default=129, help='data node number') # france 81, italy 105, spain 34, england 129 california 58 massachusetts 14
    parser.add_argument('--test_data_path', type=str, default='england_labels.csv', help='test data file used in zero shot forecasting')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=7, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=7, help='label length')
    parser.add_argument('--token_len', type=int, default=7, help='token length')
    parser.add_argument('--test_seq_len', type=int, default=7, help='test seq len')
    parser.add_argument('--test_label_len', type=int, default=7, help='test label len')
    parser.add_argument('--test_pred_len', type=int, default=7, help='test pred len')

    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    # GPU
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    args, unknown = parser.parse_known_args()
    main(args)