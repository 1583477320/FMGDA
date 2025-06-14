#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse
import torch
from optim.client_optim import METHODS

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--local_epochs', type=int, default=5, help="rounds of local training epochs")
    parser.add_argument('--global_epochs', type=int, default=40, help="rounds of global training epochs")
    parser.add_argument('--num_clients', type=int, default=5, help="number of client: K")
    parser.add_argument('--num_tasks', type=int, default=2, help="number of task: i")
    parser.add_argument('--batch_size', type=int, default=256, help="batch size: [a,b]")
    parser.add_argument('--local_lr', type=float, default=0.01, help="local_learning rate")
    parser.add_argument('--global_lr', type=float, default=0.01, help="global_learning rate")
    parser.add_argument('--beta', type=float, default=0.5, help="FMGDA_S beta (default: 0.5)")
    parser.add_argument("--method", type=str, default="fmgda_s",choices=list(METHODS.keys()), help="MTL weight method")

    # model arguments
    parser.add_argument('--kernel_num', type=int, default=3, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 1)')
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')
    args = parser.parse_args()
    return args


# 上批次客户端梯度聚合初始化
def last_client_init(client_model):
    device = torch.device(
        'cuda:{}'.format(args_parser().gpu) if torch.cuda.is_available() and args_parser().gpu != -1 else 'cpu')
    for param in client_model.parameters():
        param.grad = torch.ones_like(param.data)
    shared_parameters = list((p for n, p in client_model.named_parameters() if "task" not in n))

    grad_dims = []
    for param in shared_parameters:
        if hasattr(param, 'data'):
            grad_dims.append(param.data.numel())
    grads = torch.Tensor(sum(grad_dims), args_parser().num_tasks).to(device)

    for i in range(args_parser().num_tasks):
        grad2vec(shared_parameters, grads, grad_dims, int(i))

    return grads

@staticmethod
def grad2vec(shared_params, grads, grad_dims, task):
    '''
    梯度展平储存的函数
    '''
    # store the gradients
    # 初始化当前任务梯度列为0
    grads[:, task].fill_(0.0)
    cnt = 0
    # for mm in m.shared_modules():
    #     for p in mm.parameters():

    for param in shared_params:
        grad = param.grad
        if grad is not None:
            # 复制当前参数的梯度
            grad_cur = grad.data.detach().clone()

            # 计算在展平向量中的位置
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[: cnt + 1])

            # 将梯度展平存储到指定位置
            grads[beg:en, task].copy_(grad_cur.data.view(-1))
        cnt += 1

def overwrite_grad(shared_parameters, newgrad, grad_dims):
    # 缩放梯度以匹配总损失
    cnt = 0

    # for mm in m.shared_modules():
    #     for param in mm.parameters():
    for param in shared_parameters:
        # 计算当前参数在向量中的位置
        beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
        en = sum(grad_dims[: cnt + 1])

        # 提取对应位置的梯度并重塑为参数形状
        this_grad = newgrad[beg:en].contiguous().view(param.data.size())

        # 覆盖参数的梯度
        param.grad = this_grad.data.clone()
        cnt += 1