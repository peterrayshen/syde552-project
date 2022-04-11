import os
import h5py
import pandas as pd
import sys

import numpy as np

import torch
import torch.nn as nn
import torchvision
from torch.utils import data
import pickle
import argparse

from utils import SurrGradSpike, get_shd_dataset, sparse_data_generator_from_hdf5_spikes, sparse_data_generator_from_hdf5_spikes_2
from collections import Counter
from sklearn.model_selection import train_test_split
from constants import xp_type_tau_constant, xp_type_tau_gauss, xp_type_tau_uniform, nb_inputs, nb_hidden, nb_outputs, time_step, nb_steps, max_time, batch_size, tau_mem_readout, tau_syn

# general xp params
nb_epochs = 10
lr = 1e-3
num_trials = 2

for xp_type in [xp_type_tau_uniform, xp_type_tau_gauss, xp_type_tau_constant]: 
    if xp_type == xp_type_tau_uniform:
        xp_params = {
            'uniform_lower_bounds': np.array([38, 35, 30, 20]) * 1e-3,
            'uniform_upper_bounds':  np.array([42, 45, 50, 60]) * 1e-3,
            'num_xps': 4,
        }
    elif xp_type == xp_type_tau_gauss:
        xp_params = {
            'gauss_mean': np.array([40, 40, 40, 40]) * 1e-3,
            'gauss_std':  np.array([2, 5, 10, 20]) * 1e-3,
            'num_xps': 4,
        }
    elif xp_type == xp_type_tau_constant:
        xp_params = {
            'constant_tau': np.array([20, 40, 60, 80]) * 1e-3,
            'num_xps': 4,
        }
    else:
        raise ValueError('xp type not recognized')


    dtype = torch.float

    # Check whether a GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")     
    else:
        device = torch.device("cpu")
    print(f'Using device: {device}')

    train_file = h5py.File('data/shd_train.h5', 'r')
    test_file = h5py.File('data/shd_test.h5', 'r')

    x_train = train_file['spikes']
    y_train = train_file['labels']
    x_test = test_file['spikes']
    y_test = test_file['labels']

    x_train_df = pd.DataFrame()
    x_train_df['times'] = np.array(x_train['times'])
    x_train_df['units'] = np.array(x_train['units'])
    y_train_np = np.array(y_train)

    x_train_train, x_train_valid, y_train_train, y_train_valid = train_test_split(x_train_df, y_train_np, test_size=0.2, random_state=42, stratify=y_train_np)

    alpha   = float(np.exp(-time_step/tau_syn))
    beta_readout    = float(np.exp(-time_step/tau_mem_readout))

    xp_results = []

    batch_cache_train_train = list(sparse_data_generator_from_hdf5_spikes_2(x_train_train, y_train_train, batch_size, nb_steps, nb_inputs, max_time, device))
    batch_cache_train_valid = list(sparse_data_generator_from_hdf5_spikes_2(x_train_valid, y_train_valid, batch_size, nb_steps, nb_inputs, max_time, device))
    batch_cache_test = list(sparse_data_generator_from_hdf5_spikes(x_test, y_test, batch_size, nb_steps, nb_inputs, max_time, device, shuffle=False))


    def get_mini_batch(x_data, y_data, shuffle=False):
        for ret in sparse_data_generator_from_hdf5_spikes(x_data, y_data, batch_size, nb_steps, nb_inputs, max_time, device, shuffle=shuffle):
            return ret 
    x_batch_mini, _ = get_mini_batch(x_test, y_test)

    for trial_num in range(num_trials):
        for i in range(xp_params['num_xps']):
            if xp_type == xp_type_tau_uniform:
                uniform_lower = xp_params['uniform_lower_bounds'][i]
                uniform_upper = xp_params['uniform_upper_bounds'][i]
                gen_tau_np = np.tile(np.random.uniform(low=uniform_lower, high=uniform_upper, size=nb_hidden), (batch_size, 1))
                print(f"init done for lower bound {uniform_lower}, upper bound {uniform_upper}")
            elif xp_type == xp_type_tau_gauss:
                gauss_mean = xp_params['gauss_mean'][i]
                gauss_std = xp_params['gauss_std'][i]
                gen_tau_np = np.tile(np.random.normal(loc=gauss_mean, scale=gauss_std, size=(nb_hidden)), (batch_size, 1))
                print(f"init done for mean {gauss_mean}, stddev {gauss_std}")
            elif xp_type == xp_type_tau_constant:
                constant_tau = xp_params['constant_tau'][i]
                gen_tau_np = np.ones((batch_size,nb_hidden)) * constant_tau
                print(f"init done for constant tau {constant_tau}")
            else:
                raise ValueError('xp type not recognized')

            gen_tau_np[gen_tau_np < 5e-3] = 5e-3
            beta_np = np.exp(-time_step/gen_tau_np)
            beta_torch = torch.from_numpy(beta_np).float().to(device=device)

            weight_scale = 0.2

            w1 = torch.empty((nb_inputs, nb_hidden),  device=device, dtype=dtype, requires_grad=True)
            torch.nn.init.normal_(w1, mean=0.0, std=weight_scale/np.sqrt(nb_inputs))

            w2 = torch.empty((nb_hidden, nb_outputs), device=device, dtype=dtype, requires_grad=True)
            torch.nn.init.normal_(w2, mean=0.0, std=weight_scale/np.sqrt(nb_hidden))

            v1 = torch.empty((nb_hidden, nb_hidden), device=device, dtype=dtype, requires_grad=True)
            torch.nn.init.normal_(v1, mean=0.0, std=weight_scale/np.sqrt(nb_hidden))

            spike_fn  = SurrGradSpike.apply

            def run_snn(inputs):
                syn = torch.zeros((batch_size,nb_hidden), device=device, dtype=dtype)
                mem = torch.zeros((batch_size,nb_hidden), device=device, dtype=dtype)

                mem_rec = []
                spk_rec = []

                # Compute hidden layer activity
                out = torch.zeros((batch_size, nb_hidden), device=device, dtype=dtype)
                h1_from_input = torch.einsum("abc,cd->abd", (inputs, w1))
                for t in range(nb_steps):
                    h1 = h1_from_input[:,t] + torch.einsum("ab,bc->ac", (out, v1))
                    mthr = mem-1.0
                    out = spike_fn(mthr)
                    rst = out.detach() # We do not want to backprop through the reset

                    new_syn = alpha*syn +h1
                    new_mem =(beta_torch*mem +syn)*(1.0-rst)

                    mem_rec.append(mem)
                    spk_rec.append(out)
                    
                    mem = new_mem
                    syn = new_syn

                mem_rec = torch.stack(mem_rec,dim=1)
                spk_rec = torch.stack(spk_rec,dim=1)

                # Readout layer
                h2= torch.einsum("abc,cd->abd", (spk_rec, w2))
                flt = torch.zeros((batch_size,nb_outputs), device=device, dtype=dtype)
                out = torch.zeros((batch_size,nb_outputs), device=device, dtype=dtype)
                out_rec = [out]
                for t in range(nb_steps):
                    new_flt = alpha*flt +h2[:,t]
                    new_out = beta_readout*out +flt

                    flt = new_flt
                    out = new_out

                    out_rec.append(out)

                out_rec = torch.stack(out_rec,dim=1)
                other_recs = [mem_rec, spk_rec]
                return out_rec, other_recs

            def compute_classification_accuracy(batch_cache):
                """ Computes classification accuracy on supplied data in batches. """
                accs = []
                for x_local, y_local in batch_cache_test:
                    output,_ = run_snn(x_local.to_dense())
                    m,_= torch.max(output,1) # max over time
                    _,am=torch.max(m,1)      # argmax over output units
                    tmp = np.mean((y_local==am).detach().cpu().numpy()) # compare to labels
                    accs.append(tmp)
                return np.mean(accs)

            def compute_classification_accuracy_2(batch_cache, device):
                """ Computes classification accuracy on supplied data in batches. """
                accs = []
                for x_local, y_local in batch_cache:
                    output,_ = run_snn(x_local.to_dense())
                    m,_= torch.max(output,1) # max over time
                    _,am=torch.max(m,1)      # argmax over output units
                    tmp = np.mean((y_local==am).detach().cpu().numpy()) # compare to labels
                    accs.append(tmp)
                return np.mean(accs)

            def train_with_validation(
                x_train_train, 
                y_train_train,
                x_train_valid,
                y_train_valid, 
                lr=1e-3, 
                nb_epochs=10,
                ):
                params = [w1,w2,v1]
                optimizer = torch.optim.Adamax(params, lr=lr, betas=(0.9,0.999))

                log_softmax_fn = nn.LogSoftmax(dim=1)
                loss_fn = nn.NLLLoss()
                
                loss_train_supervised = []
                loss_train_reg = []
                loss_valid_supervised = []
                loss_valid_reg = []
                acc_train = []
                acc_valid = []

                for e in range(nb_epochs):
                    # train
                    local_loss_train_supervised = []
                    local_loss_train_reg = []
                    for x_local, y_local in batch_cache_train_train:
                        output,recs = run_snn(x_local.to_dense())
                        _,spks=recs
                        m,_=torch.max(output,1)
                        log_p_y = log_softmax_fn(m)
                        
                        # Here we set up our regularizer loss
                        # The strength paramters here are merely a guess and there should be ample room for improvement by
                        # tuning these paramters.
                        reg_loss = 2e-6*torch.sum(spks) # L1 loss on total number of spikes
                        reg_loss += 2e-6*torch.mean(torch.sum(torch.sum(spks,dim=0),dim=0)**2) # L2 loss on spikes per neuron
                        
                        # Here we combine supervised loss and the regularizer
                        loss_val = loss_fn(log_p_y, y_local) + reg_loss

                        optimizer.zero_grad()
                        loss_val.backward()
                        optimizer.step()
                        local_loss_train_supervised.append(loss_fn(log_p_y, y_local).item())
                        local_loss_train_reg.append(reg_loss.item())
                    mean_supervised_loss = np.mean(local_loss_train_supervised)
                    mean_valid_loss = np.mean(local_loss_train_reg)
                    loss_train_supervised.append(mean_supervised_loss)
                    loss_train_reg.append(mean_valid_loss)
                    local_acc_train = compute_classification_accuracy_2(batch_cache_train_train, device)
                    acc_train.append(local_acc_train)
                    print("Epoch %i: train loss=%.5f, train acc=%.5f"%(e+1,mean_supervised_loss + mean_valid_loss, local_acc_train))

                    # validation
                    local_loss_valid_supervised = []
                    local_loss_valid_reg = []
                    for x_local, y_local in batch_cache_train_valid:
                        output,recs = run_snn(x_local.to_dense())
                        _,spks=recs
                        m,_=torch.max(output,1)
                        log_p_y = log_softmax_fn(m)
                        
                        # Here we set up our regularizer loss
                        # The strength paramters here are merely a guess and there should be ample room for improvement by
                        # tuning these paramters.
                        reg_loss = 2e-6*torch.sum(spks) # L1 loss on total number of spikes
                        reg_loss += 2e-6*torch.mean(torch.sum(torch.sum(spks,dim=0),dim=0)**2) # L2 loss on spikes per neuron
                        
                        local_loss_valid_supervised.append(loss_fn(log_p_y, y_local).item())
                        local_loss_valid_reg.append(reg_loss.item())
                    mean_supervised_loss = np.mean(local_loss_valid_supervised)
                    mean_valid_loss = np.mean(local_loss_valid_reg)
                    loss_valid_supervised.append(mean_supervised_loss)
                    loss_valid_reg.append(mean_valid_loss)
                    local_acc_valid = compute_classification_accuracy_2(batch_cache_train_train, device)
                    acc_valid.append(local_acc_valid)
                    print("Epoch %i: validation loss=%.5f, validation acc=%.5f"%(e+1,mean_supervised_loss + mean_valid_loss, local_acc_valid))
                
                return loss_train_supervised, loss_valid_supervised, loss_train_reg, loss_valid_reg, acc_train, acc_valid

            loss_supervised_train, loss_supervised_valid, loss_reg_train, loss_reg_valid, acc_train, acc_valid = train_with_validation(x_train_train, y_train_train, x_train_valid, y_train_valid, lr=lr, nb_epochs=nb_epochs)

            output, other_recordings = run_snn(x_batch_mini.to_dense())
            mem_rec, spk_rec = other_recordings
            acc_test = compute_classification_accuracy(batch_cache_test)

            xp_res = {
                'w1': w1.cpu().detach().numpy(),
                'w2': w2.cpu().detach().numpy(),
                'v1': v1.cpu().detach().numpy(),
                'weight_scale': weight_scale,
                'xp_type': xp_type,
                'xp_params': xp_params,
                'i': i,
                'gen_distribution_tau': gen_tau_np,
                'gen_distribution_beta': beta_np,
                'alpha': alpha,
                'beta_readout': beta_readout,
                'nb_inputs': nb_inputs,
                'nb_hidden': nb_hidden,
                'nb_outputs': nb_outputs,
                'time_step': time_step,
                'nb_steps': nb_steps,
                'max_time': max_time,
                'batch_size': batch_size,
                'tau_mem_readout': tau_mem_readout,
                'tau_syn': tau_syn,
                'lr': lr,
                'loss_supervised_train': loss_supervised_train,
                'loss_supervised_valid': loss_supervised_valid,
                'loss_reg_train': loss_reg_train,
                'loss_reg_valid': loss_reg_valid,
                'acc_test': acc_test,
                'acc_train': acc_train,
                'acc_valid': acc_valid,
                'output': output.cpu().detach().numpy(),
                'mem_rec': mem_rec.cpu().detach().numpy(),
                'spk_rec': spk_rec.cpu().detach().numpy(),
                'trial_num': trial_num,
            }
            xp_results.append(xp_res)

    with open(f'results/learn_curve_{xp_type}.pkl', 'wb') as f:
        pickle.dump(xp_results, f)


