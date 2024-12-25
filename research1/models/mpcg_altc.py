#  三分类掩码式模型
import torch.nn as nn
import torch
from transformers.models.bert.modeling_bert import BertSelfAttention, BertPooler
from transformers.models.roberta.modeling_roberta import RobertaSelfAttention, RobertaPooler
from transformers.models.bart.modeling_bart import BartAttention
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
import copy
import torch.nn.functional as F


class MyBartPooler(nn.Module):
    def __init__(self, dim):
        super(MyBartPooler, self).__init__()
        self.dense = nn.Linear(dim, dim)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        last_token_tensor = hidden_states[:, -1]
        pooled_output = self.dense(last_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class CrossEntropyLoss_LSR(nn.Module):
    def __init__(self, device, para_LSR=0.2):
        super(CrossEntropyLoss_LSR, self).__init__()
        self.para_LSR = para_LSR
        self.device = device
        self.logSoftmax = nn.LogSoftmax(dim=-1)

    def _toOneHot_smooth(self, label, batchsize, classes):
        prob = self.para_LSR * 1.0 / classes
        one_hot_label = torch.zeros(batchsize, classes) + prob
        for i in range(batchsize):
            index = label[i]
            one_hot_label[i, index] += (1.0 - self.para_LSR)
        return one_hot_label

    def forward(self, pre, label, size_average=True):
        b, c = pre.size()
        one_hot_label = self._toOneHot_smooth(label, b, c).to(self.device)
        #loss = torch.sum(-one_hot_label * self.logSoftmax(pre), dim=1)
        loss = torch.sum(-one_hot_label * pre, dim=1)
        if size_average:
            return torch.mean(loss)
        else:
            return torch.sum(loss)


class MyAveragePool(nn.Module):
    def __init__(self, dim):
        super(MyAveragePool, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        result = inputs.mean(dim=self.dim)
        return result


# prompt_csr
class PROMPT_MPCG_ALTC(nn.Module):
    def __init__(self, premodel, opt):
        super(PROMPT_MPCG_ALTC, self).__init__()
        self.opt = opt
        self.encode_bert_gloabl = copy.deepcopy(premodel)
        if opt.pretrained_model == 'bert-base':
            self.mha = BertSelfAttention(premodel.config)
        elif opt.pretrained_model == 'roberta-base':
            self.mha = RobertaSelfAttention(premodel.config)
        elif opt.pretrained_model == 'bart-base':
            self.mha = BartAttention(premodel.config.d_model, premodel.config.encoder_attention_heads, premodel.config.dropout)
        elif opt.pretrained_model == 'gpt2':
            self.mha = GPT2Attention(premodel.config)
        else:
            print('error in pre-trained model!')
            exit(0)

        self.conv1 = nn.Conv2d(opt.bert_dim, opt.hidden_dim, kernel_size=(1, 1))
        self.convn = nn.ModuleList(
            [nn.Conv2d(opt.hidden_dim, opt.hidden_dim, kernel_size=(1, 1)) for _ in range(opt.mpc-2)])
        self.convm = nn.Conv2d(opt.hidden_dim, opt.bert_dim, kernel_size=(1, 1))

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, int(opt.bert_dim / 3), (k, opt.bert_dim)) for k in self.opt.kernel_size])
        self.maxs = nn.ModuleList(
            [nn.MaxPool1d(kernel_size=opt.max_seq_len - k + 1) for k in self.opt.kernel_size])

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(opt.dropout)
        self.pooling = MyAveragePool(dim=1)

        self.concat_double_linear = nn.Linear(opt.bert_dim * 2, opt.bert_dim)

        self.gate = nn.Sequential(
            nn.Linear(opt.bert_dim * 2, opt.bert_dim * 2),
            nn.Sigmoid()
        )
        self.classifier_2 = nn.Linear(opt.bert_dim * 2, opt.polarities_dim)
        self.classifier_3 = nn.Linear(opt.bert_dim * 3, opt.polarities_dim)
        self.classifier_1 = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def _getMask(self, input, mask_idx):
        '''
        :param input:  shape:b,n,d
        :param mask_idx: b
        :return: b,d
        '''
        new_input = None
        for idx in range(input.size(0)):
            if new_input == None:
                new_input = input[idx][mask_idx[idx]].unsqueeze(dim=0)
            else:
                new_input = torch.cat((new_input, input[idx][mask_idx[idx]].unsqueeze(dim=0)), dim=0)
        return new_input

    def forward(self, inputs):
        if self.opt.pretrained_model != 'gpt2':
            concat_input_ids, concat_token_type_ids, mask_idx = inputs[0], inputs[1], inputs[2]
            bert_global_embedding = self.encode_bert_gloabl(input_ids=concat_input_ids)['last_hidden_state']
        else:
            concat_input_ids, concat_token_type_ids, mask_idx, attention_mask = inputs[0], inputs[1], inputs[2], inputs[3]
            bert_global_embedding = self.encode_bert_gloabl(
                input_ids=concat_input_ids, attention_mask=attention_mask)['last_hidden_state']

        # pct全局语义细化
        cnn_out1 = self.act(
            self.conv1(bert_global_embedding.unsqueeze(-1).transpose(1, 2)))  # b,n,d,1->b,d,n,1->b,x,n,1
        if len(self.convn):
            cnn_out = cnn_out1
            for con in self.convn:
                cnn_out = self.act(con(cnn_out))
            cnn_outm = self.act(self.convm(cnn_out)).transpose(1, 2).squeeze(-1)  # b,x,n,1->b,d,n,1->b,n,d,1
        else:
            cnn_outm = self.act(self.convm(cnn_out1)).transpose(1, 2).squeeze(-1)  # b,x,n,1->b,d,n,1->b,n,d,1
        global_out = self.dropout(cnn_outm)

        # 语义融合 残差式 or 拼接式
        if self.opt.residual:
            global_output = global_out + bert_global_embedding
            # global_output = nn.LayerNorm(global_out.shape).to(self.opt.device)(global_output)
        else:
            concat_out = self.concat_double_linear(torch.cat([bert_global_embedding, global_out], -1))
            concat_out = self.mha(concat_out)[0]
            global_output = self.dropout(concat_out)


        # 标准cnn提取局部语义
        local_out = None
        for conv, max in zip(self.convs, self.maxs):
            conv_input = bert_global_embedding.unsqueeze(1)  # (B,N,D)->(B,1,N,D)
            conv_output = self.act(conv(conv_input).squeeze(-1))  # (B,1,N,D)->(B,O_C,N-K+1)
            max_output = max(conv_output).squeeze(-1)  # (B,O_C,N-K+1)->(B,O_C)
            if local_out is None:
                local_out = max_output
            else:
                local_out = torch.cat([local_out, max_output], -1)

        # 提取MASK所在位置
        mask_out = self._getMask(bert_global_embedding, mask_idx)

        if self.opt.pool:
            pool_out = self.pooling(global_output)
            pool_mask_out = self.gate(torch.cat([mask_out, pool_out], -1))
            concat_out = torch.cat([pool_mask_out, local_out], -1)
            output = self.classifier_3(concat_out)
        else:
            concat_out = torch.cat([mask_out, local_out], -1)
            output = self.classifier_2(concat_out)

        return F.log_softmax(output, dim=-1)


# w/o csr
class PROMPT_BASE(nn.Module):
    def __init__(self, pre_model, opt):
        super(PROMPT_BASE, self).__init__()
        self.opt = opt
        self.encode = pre_model
        self.classifier = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def _getMask(self, input, mask_idx):
        '''
        :param input:  shape:b,n,d
        :param mask_idx: b
        :return: b,d
        '''
        new_input = None
        for idx in range(input.size(0)):
            if new_input == None:
                new_input = input[idx][mask_idx[idx]].unsqueeze(dim=0)
            else:
                new_input = torch.cat((new_input, input[idx][mask_idx[idx]].unsqueeze(dim=0)), dim=0)
        return new_input

    def forward(self, inputs):
        if self.opt.pretrained_model != 'gpt2':
            concat_input_ids, concat_token_type_ids, mask_idx = inputs[0], inputs[1], inputs[2]
            embedding = self.encode(input_ids=concat_input_ids)[0]
        else:
            concat_input_ids, concat_token_type_ids, mask_idx, attention_mask = inputs[0], inputs[1], inputs[2], inputs[3]
            embedding = self.encode(input_ids=concat_input_ids, attention_mask = attention_mask)[0]

        mask_out = self._getMask(embedding, mask_idx)
        output = self.classifier(mask_out)
        return F.log_softmax(output, dim=-1)


# w/o prompt
class MPCG_ALTC(nn.Module):
    def __init__(self, premodel, opt):
        super(MPCG_ALTC, self).__init__()
        self.opt = opt
        self.encode_bert_gloabl = premodel
        if opt.pretrained_model == 'bert-base':
            self.mha = BertSelfAttention(premodel.config)
            self.pooler = BertPooler(premodel.config)
        elif opt.pretrained_model == 'roberta-base':
            self.mha = RobertaSelfAttention(premodel.config)
            self.pooler = RobertaPooler(premodel.config)
        elif opt.pretrained_model == 'bart-base':
            self.mha = self.mha = BartAttention(premodel.config.d_model, premodel.config.encoder_attention_heads, premodel.config.dropout)
            self.pooler = MyBartPooler(dim=opt.bert_dim)
        else:
            print('error in pre-trained model!')
            exit(0)

        self.conv1 = nn.Conv2d(opt.bert_dim, opt.hidden_dim, kernel_size=(1, 1))
        self.convn = nn.ModuleList(
            [nn.Conv2d(opt.hidden_dim, opt.hidden_dim, kernel_size=(1, 1)) for _ in range(opt.mpc-2)])
        self.convm = nn.Conv2d(opt.hidden_dim, opt.bert_dim, kernel_size=(1, 1))

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, int(opt.bert_dim / 3), (k, opt.bert_dim)) for k in self.opt.kernel_size])
        self.maxs = nn.ModuleList(
            [nn.MaxPool1d(kernel_size=opt.max_seq_len - k + 1) for k in self.opt.kernel_size])

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(opt.dropout)
        self.pooling = MyAveragePool(dim=1)

        self.concat_double_linear = nn.Linear(opt.bert_dim * 2, opt.bert_dim)
        self.gate = nn.Sequential(
            nn.Linear(opt.bert_dim * 2, opt.bert_dim * 2),
            nn.Sigmoid()
        )
        self.classifier_2 = nn.Linear(opt.bert_dim * 2, opt.polarities_dim)
        self.classifier_3 = nn.Linear(opt.bert_dim * 3, opt.polarities_dim)
        self.classifier_1 = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
        concat_input_ids, concat_token_type_ids = inputs[0], inputs[1]

        # embedding BERT编码
        bert_global_embedding = self.encode_bert_gloabl(input_ids=concat_input_ids)['last_hidden_state']

        # pct全局语义细化
        cnn_out1 = self.act(
            self.conv1(bert_global_embedding.unsqueeze(-1).transpose(1, 2)))  # b,n,d,1->b,d,n,1->b,x,n,1
        if len(self.convn):
            cnn_out = cnn_out1
            for con in self.convn:
                cnn_out = self.act(con(cnn_out))
            cnn_outm = self.act(self.convm(cnn_out)).transpose(1, 2).squeeze(-1)  # b,x,n,1->b,d,n,1->b,n,d,1
        else:
            cnn_outm = self.act(self.convm(cnn_out1)).transpose(1, 2).squeeze(-1)  # b,x,n,1->b,d,n,1->b,n,d,1
        global_out = self.dropout(cnn_outm)

        # 语义融合 残差式 or 拼接式
        if self.opt.residual:
            global_output = global_out + bert_global_embedding
            # global_output = nn.LayerNorm(global_out.shape).to(self.opt.device)(global_output)
        else:
            concat_out = self.concat_double_linear(torch.cat([bert_global_embedding, global_out], -1))
            concat_out = self.mha(concat_out)[0]
            global_output = self.dropout(concat_out)


        # 标准cnn提取局部语义
        local_out = None
        for conv, max in zip(self.convs, self.maxs):
            conv_input = bert_global_embedding.unsqueeze(1)  # (B,N,D)->(B,1,N,D)
            conv_output = self.act(conv(conv_input).squeeze(-1))  # (B,1,N,D)->(B,O_C,N-K+1)
            max_output = max(conv_output).squeeze(-1)  # (B,O_C,N-K+1)->(B,O_C)
            if local_out is None:
                local_out = max_output
            else:
                local_out = torch.cat([local_out, max_output], -1)

        pool_out = self.pooling(global_output)
        pooler_out = self.pooler(global_output)
        pool_output = self.gate(torch.cat([pooler_out, pool_out], -1))
        concat_out = torch.cat([pool_output, local_out], -1)
        output = self.classifier_3(concat_out)
        return F.log_softmax(output, dim=-1)