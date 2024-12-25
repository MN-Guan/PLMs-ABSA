# -*- coding: utf-8 -*-
# file: train.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.
import logging
import argparse
import math
import os
import sys
import pandas as pd
import random
from distutils.util import strtobool
import numpy as np
from time import strftime, localtime
from transformers import BertModel, BartModel, RobertaModel, GPT2Model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert, PromptDataset, ABSADataset
from models.mpcg_altc import PROMPT_BASE
from models.mpcg_altc import CrossEntropyLoss_LSR
from transformers import logging as tran_logging
from sklearn.metrics import precision_recall_fscore_support as macro_f1

tran_logging.set_verbosity_warning()
tran_logging.set_verbosity_error()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

pre_train_map = {
    'bart-base': {
        'tokenizer_file': 'model/bart-base/',
        'pre_model_file': 'model/bart-base'
    },
    'roberta-base': {
        'tokenizer_file': 'model/roberta-base/',
        'pre_model_file': 'model/roberta-base'
    },
    'bert-base':
        {
        'tokenizer_file': 'model/uncased_L-12_H-768_A-12/',
        'pre_model_file': 'model/uncased_L-12_H-768_A-12'
        },
    'gpt2':
        {
        'tokenizer_file': 'model/gpt2/',
        'pre_model_file': 'model/gpt2'
        }
}


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        tokenizer = Tokenizer4Bert(
            opt.max_seq_len, pre_train_map[opt.pretrained_model]['tokenizer_file'], opt.pretrained_model)
        if opt.pretrained_model == 'bart-base':
            pre_model = BartModel.from_pretrained(pre_train_map[opt.pretrained_model]['pre_model_file'])
        elif opt.pretrained_model == 'roberta-base':
            pre_model = RobertaModel.from_pretrained(pre_train_map[opt.pretrained_model]['pre_model_file'])
        elif opt.pretrained_model == 'bert-base':
            pre_model = BertModel.from_pretrained(pre_train_map[opt.pretrained_model]['pre_model_file'])
        elif opt.pretrained_model == 'gpt2':
            pre_model = GPT2Model.from_pretrained(pre_train_map[opt.pretrained_model]['pre_model_file'])
        else:
            print('error in your pretrained_model!')
            exit(0)
        self.model = opt.model_class(pre_model, opt).to(opt.device)

        if opt.model_name == 'prompt_mpcg_altc':
            self.trainset = PromptDataset(
                opt.pretrained_model, opt.dataset_file['train'], tokenizer, opt.max_seq_len,
                opt.dataset, opt.twitter_set, opt.wo_keyword, 'train')
            self.testset = PromptDataset(
                opt.pretrained_model, opt.dataset_file['test'], tokenizer, opt.max_seq_len,
                opt.dataset, opt.twitter_set, opt.wo_keyword, 'test')
        elif opt.model_name == 'mpcg_altc':
            self.trainset = ABSADataset(opt.dataset_file['train'], tokenizer, opt.max_seq_len)
            self.testset = ABSADataset(opt.dataset_file['test'], tokenizer, opt.max_seq_len)
        elif opt.model_name == 'prompt_base':
            self.trainset = PromptDataset(opt.pretrained_model, opt.dataset_file['train'], tokenizer, opt.max_seq_len,
                                          opt.dataset, opt.twitter_set)
            self.testset = PromptDataset(opt.pretrained_model, opt.dataset_file['test'], tokenizer, opt.max_seq_len,
                                         opt.dataset, opt.twitter_set, type='test')
        else:
            print('error in your model_name!')
            exit(0)

        assert 0 <= opt.valset_ratio < 1
        if opt.valset_ratio > 0:  # 划分训练集和验证集**************************
            valset_len = int(len(self.trainset) * opt.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset) - valset_len, valset_len))
        else:
            self.valset = self.testset

        logger.info(
            'trainset: {}\tvalset: {}\ttestset: {}'.format(len(self.trainset), len(self.valset), len(self.testset)))
        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info(
            '> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        # 如果不是bert模型需要调整someting
        for child in self.model.children():  # model的大框架
            if type(child) not in [BertModel, RobertaModel, BartModel, GPT2Model]:
                print('{}:调整'.format(type(child)))
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
        # 训练
        max_val_acc = 0   # 记录最大正确率
        max_val_f1 = 0   # 记录最大F1
        max_val_epoch = 0  # 记录最佳epoch
        global_step = 0
        path = None

        for i_epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(i_epoch + 1))
            n_total, loss_total = 0, 0  # 记录一个epoch中总样本数和总的损失
            # switch model to training mode
            self.model.train()
            for i_batch, batch in enumerate(train_data_loader):
                global_step += 1
                # 梯度清零
                optimizer.zero_grad()

                inputs = [batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                outputs = self.model(inputs)
                targets = batch['label'].to(self.opt.device)

                loss = criterion(outputs, targets)  # 这里的梯度是一个batch_size的平均梯度
                loss.backward()
                optimizer.step()

                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)

                if global_step % self.opt.log_step == 0:
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}'.format(train_loss))

            val_correct, val_total, val_acc, val_f1, nc_dict = self._evaluate_acc_f1(val_data_loader)
            logger.info('> val_acc: {}/{}={:.4f}, val_f1: {:.4f}'.format(val_correct, val_total, val_acc, val_f1))
            logger.info('消极:{}/{}={:.4f},中性:{}/{}={:.4f},积极:{}/{}={:.4f}'.format(
                nc_dict['0'][0], nc_dict['0'][1], nc_dict['0'][0] / nc_dict['0'][1],
                nc_dict['1'][0], nc_dict['1'][1], nc_dict['1'][0] / nc_dict['1'][1],
                nc_dict['2'][0], nc_dict['2'][1], nc_dict['2'][0] / nc_dict['2'][1], ))

            if (val_acc > max_val_acc) or (val_acc == max_val_acc and val_f1 > max_val_f1):
                max_val_acc = val_acc
                max_val_epoch = i_epoch
                max_val_f1 = val_f1
                if not os.path.exists('state_dict'):
                    os.mkdir('state_dict')
                path = 'state_dict/{0}_{1}_val_acc_{2}'.format(self.opt.model_name, self.opt.dataset, round(val_acc, 4))
                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))

            if i_epoch - max_val_epoch >= self.opt.patience:
                print('>> early stop.')
                break
        return path

    def _evaluate_acc_f1(self, data_loader, type='dev'):
        # 测试
        n_correct, n_total, n_f1 = 0, 0, 0
        nc_dict = {
            '0': [0, 0],  # 消极
            '1': [0, 0],  # 中性
            '2': [0, 0]  # 积极
        }
        t_outputs_all, labels_all, t_outputs_max_all = None, None, None
        self.model.eval()
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                t_inputs = [t_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_outputs = self.model(t_inputs)  # 输出
                t_outputs_argmax = torch.argmax(t_outputs, -1)  # 最大概率索引
                labels = t_batch['label']

                if t_outputs_all is None:
                    t_outputs_all = t_outputs.cpu()  # 模型输出
                    labels_all = labels  # 样本标签
                    t_outputs_max_all = t_outputs_argmax  # 概率大的标签
                else:
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs.cpu()), dim=0)
                    labels_all = torch.cat((labels_all, labels), dim=0)
                    t_outputs_max_all = torch.cat((t_outputs_max_all, t_outputs_argmax), dim=0)

            for pre, tat in zip(t_outputs_max_all.cpu().numpy(), labels_all.cpu().numpy()):
                n_total += 1
                nc_dict[str(tat)][1] += 1
                if pre == tat:
                    nc_dict[str(tat)][0] += 1
                    n_correct += 1

            acc = n_correct / n_total
            # print(labels_all)
            # print(t_outputs_max_all)
            p_class, r_class, f_class, support_micro = macro_f1(labels_all.cpu(), t_outputs_max_all.cpu(),
                                                                labels=[0, 1, 2], average=None)

            logger.info('F1: {} -- Mean F1: {}'.format(f_class, f_class.mean()))
            logger.info('ACC: {}'.format(acc))
            if type == 'test':
                self._write_detail(acc, f_class.mean(), nc_dict, f_class)
            return n_correct, n_total, acc, f_class.mean(), nc_dict

    def _write_detail(self, test_acc, test_f1, acc_detail, f1_detail):
        result_file = 'log/{}/{}/{}/result_{}.csv'.format(self.opt.log_date, self.opt.pretrained_model, self.opt.dataset, self.opt.dataset)
        if not os.path.exists(result_file):
            title = ['数据集', '模型', 'residual', 'kernel_size',
                     '消极ACC', '消极F1', '中性ACC', '中性F1', '积极ACC', '积极F1', 'ACC', 'F1']
            data = [self.opt.dataset, self.opt.model_name, self.opt.residual, self.opt.kernel_size,
                    '%.4f' % (acc_detail['0'][0] / acc_detail['0'][1]), '%.4f' % f1_detail[0],
                    '%.4f' % (acc_detail['1'][0] / acc_detail['1'][1]), '%.4f' % f1_detail[1],
                    '%.4f' % (acc_detail['2'][0] / acc_detail['2'][1]), '%.4f' % f1_detail[2],
                    '{:.4f}'.format(test_acc), '{:.4f}'.format(test_f1)]
            if self.opt.model_name != 'prompt_base':
                title.append('mpc')
                data.append(self.opt.mpc)
            df = pd.DataFrame([data], columns=title)
            df.to_csv(result_file, index=False)
        else:
            title = ['数据集', '模型', 'residual', 'kernel_size',
                     '消极ACC', '消极F1', '中性ACC', '中性F1', '积极ACC', '积极F1', 'ACC', 'F1']
            data = {
                '数据集': [self.opt.dataset],
                '模型': [self.opt.model_name],
                'residual': [self.opt.residual],
                'kernel_size': [self.opt.kernel_size],
                '消极ACC': ['%.4f' % (acc_detail['0'][0] / acc_detail['0'][1])],
                '消极F1': ['%.4f' % f1_detail[0]],
                '中性ACC': ['%.4f' % (acc_detail['1'][0] / acc_detail['1'][1])],
                '中性F1': ['%.4f' % f1_detail[1]],
                '积极ACC': ['%.4f' % (acc_detail['2'][0] / acc_detail['2'][1])],
                '积极F1': ['%.4f' % f1_detail[2]],
                'ACC': ['{:.4f}'.format(test_acc)],
                'F1': ['{:.4f}'.format(test_f1)]
            }
            if self.opt.model_name != 'prompt_base':
                title.append('mpc')
                data['mpc'] = [self.opt.mpc]
            df = pd.DataFrame(data, columns=title)
            df.to_csv(result_file, mode='a', header=False, index=False)

    def run(self):
        if self.opt.LSR:
            criterion = CrossEntropyLoss_LSR(self.opt.device)
        else:
            criterion = nn.NLLLoss().to(self.opt.device)
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.lr, weight_decay=self.opt.l2reg)

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)

        self._reset_params()
        best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader)
        self.model.load_state_dict(torch.load(best_model_path))
        test_correct, test_total, test_acc, test_f1, nc_dict = self._evaluate_acc_f1(test_data_loader, type='test')
        logger.info('>> test_acc: {}/{}={:.4f}, test_f1: {:.4f}'.format(test_correct, test_total, test_acc, test_f1))
        logger.info('消极:{}/{}={:.4f},中性:{}/{}={:.4f},积极:{}/{}={:.4f}'.format(
            nc_dict['0'][0], nc_dict['0'][1], nc_dict['0'][0] / nc_dict['0'][1],
            nc_dict['1'][0], nc_dict['1'][1], nc_dict['1'][0] / nc_dict['1'][1],
            nc_dict['2'][0], nc_dict['2'][1], nc_dict['2'][0] / nc_dict['2'][1], ))
        os.rename(self.opt.log_file, '{}_{:.4f}.log'.format(self.opt.log_file.split('.')[0], test_acc))


def main():
    # Hyper Parameters   设置参数
    parser = argparse.ArgumentParser()
    parser.register('type', 'boolean', strtobool)
    # 模型名称
    parser.add_argument('--model_name', default='prompt_mpcg_altc', type=str,
                        help='prompt-csr:prompt_mpcg_altc; w/o prompt: mpcg_altc; w/o csr: prompt_base;'
                             ' model_base:such as bert_base')
    # 数据集
    parser.add_argument('--dataset', default='restaurant', type=str, help='twitter, restaurant, laptop, res15, res16')
    # 优化函数
    parser.add_argument('--optimizer', default='adam', type=str)
    # ?
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    # 学习率
    parser.add_argument('--lr', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    # Dropout
    parser.add_argument('--dropout', default=0.1, type=float)
    # ?
    parser.add_argument('--l2reg', default=1e-5, type=float)
    # epochs
    parser.add_argument('--num_epoch', default=20, type=int, help='try larger number for non-BERT models')
    # batch_size
    parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
    #
    parser.add_argument('--log_step', default=100, type=int)
    # embedding_dim
    parser.add_argument('--embed_dim', default=300, type=int)
    # hidden_dim
    parser.add_argument('--hidden_dim', default=768, type=int)
    # bert_dim
    parser.add_argument('--bert_dim', default=768, type=int)
    # 预训练模型名称
    parser.add_argument('--pretrained_model', default='bart-base', type=str, help='bert-base,roberta-base,gpt2')
    # 序列最大长度
    parser.add_argument('--max_seq_len', default=128, type=int)
    # 情感级维度
    parser.add_argument('--polarities_dim', default=3, type=int)
    # ?
    parser.add_argument('--hops', default=3, type=int)
    # 设置耐心值：loss持续多少次不再改变就停止训练
    parser.add_argument('--patience', default=5, type=int)
    # 设备
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    # 随机种子
    parser.add_argument('--seed', default=1234, type=int, help='set seed for reproducibility')
    # 是否进行验证
    parser.add_argument('--valset_ratio', default=0.1, type=float,
                        help='set ratio between 0 and 1 for validation support')
    parser.add_argument('--residual', default='False', type='boolean', help='全局语义细化后与原始语义连接方式，残差或者拼接')

    parser.add_argument('--kernel_number', default='1', type=str,
                        help='textCNN的kernel_size:[2,3,4]/[3,4,5]/[4,5,6]/[5,6,7]')

    parser.add_argument('--LSR', default='False', type='boolean')
    # twitter数据集中性样本是否需要随机一半选择两种模板
    parser.add_argument('--twitter_set', default='False', type='boolean')

    parser.add_argument('--pool', default='True', type='boolean')

    parser.add_argument('--mpc', default=2, type=int, help='2,3,4,5 only for bert_mpcg_altc模型')

    parser.add_argument('--log_date', default='log_test')
    parser.add_argument('--wo_keyword', default='keyword', type=str, help='模板中是否包含关键词槽')
    opt = parser.parse_args()

    kernel_size = {
        '1': [2, 3, 4],
        '2': [3, 4, 5],
        '3': [4, 5, 6],
        '4': [5, 6, 7],
    }
    opt.kernel_size = kernel_size[opt.kernel_number.rstrip('\r')]
    print(opt.wo_keyword)

    if opt.seed is not None:
        # 为什么要设置随机种子？ 确保每一次随机抽样的结果一致
        # 为什么使用相同的网络结构，跑出来的效果完全不同，用的学习率，迭代次数，batch size 都是一样？
        # 固定随机数种子是非常重要的。但是如果使用的是PyTorch等框架，还要看一下框架的种子是否固定了。
        # 还有，如果用了cuda，别忘了cuda的随机数种子
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)  # 确保相同随机种子下随机抽样的结果一致 不使用cuda
        torch.cuda.manual_seed(opt.seed)  # 确保相同随机种子下随机抽样的结果一致 使用cuda
        torch.backends.cudnn.deterministic = True  # 应该可以保证每次运行网络的时候相同输入的输出是固定的
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(opt.seed)

    model_classes = {
        # 模型调用
        'prompt_mpcg_altc': PROMPT_MPCG_ALTC,
        'mpcg_altc': MPCG_ALTC,
        'prompt_base': PROMPT_BASE,
        # default hyper-parameters for LCF-BERT model is as follws:
        # lr: 2e-5
        # l2: 1e-5
        # batch size: 16
        # num epochs: 20
    }

    dataset_files = {
        # 数据集地址
        'twitter': {
            'train': '../Dataset/acl-14-short-data/train.raw',
            'test': '../Dataset/acl-14-short-data/test.raw'
        },
        'restaurant': {
            'train': '../Dataset/semeval14/Restaurants_Train.xml.seg',
            'test': '../Dataset/semeval14/Restaurants_Test_Gold.xml.seg'
        },
        'laptop': {
            'train': '../Dataset/semeval14/Laptops_Train.xml.seg',
            'test': '../Dataset/semeval14/Laptops_Test_Gold.xml.seg'
        },
        'res15': {
            'train': '../Dataset/res15/res_15_train.txt',
            'test': '../Dataset/res15/res_15_test.txt'
        },
        'res16': {
            'train': '../Dataset/res16/res_16_train.txt',
            'test': '../Dataset/res16/res_16_test.txt'
        }
    }

    input_colses = {
        'prompt_base': ['prompt_bert_indices', 'prompt_segments_indices', 'mask_idx'],
    }

    initializers = {
        # 选择初始化方式，通过网络层时，输入和输出的方差相同，包括前向传播和后向传播。
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }

    optimizers = {
        # 选择优化函数
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }

    opt.model_class = model_classes[opt.model_name]  # 选择模型种类
    opt.dataset_file = dataset_files[opt.dataset]  # 数据集路径
    opt.inputs_cols = input_colses[opt.model_name]
    if opt.pretrained_model == 'gpt2':
        opt.inputs_cols.append('attention_mask')
    opt.initializer = initializers[opt.initializer]  # 选择初始化方式
    opt.optimizer = optimizers[opt.optimizer]  # 选择优化器
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    # 日志文件
    if not os.path.exists('log'):  # 存储运行日志文件夹
        os.mkdir('log')
    if not os.path.exists('log/{}'.format(opt.log_date)):
        os.mkdir('log/{}'.format(opt.log_date))
    if not os.path.exists('log/{}/{}'.format(opt.log_date, opt.pretrained_model)):
        os.mkdir('log/{}/{}'.format(opt.log_date, opt.pretrained_model))
    if not os.path.exists('log/{}/{}/{}'.format(opt.log_date,opt.pretrained_model, opt.dataset)):
        os.mkdir('log/{}/{}/{}'.format(opt.log_date,opt.pretrained_model, opt.dataset))
    log_file = 'log/{}/{}/{}/class{}_{}-{}.log'.format(opt.log_date, opt.pretrained_model, opt.dataset, opt.polarities_dim, opt.model_name,
                                                    strftime("%y%m%d-%H:%M", localtime()))
    opt.log_file = log_file
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
