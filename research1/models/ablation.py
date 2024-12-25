import torch.nn as nn
import torch
from transformers.models.bert.modeling_bert import BertSelfAttention, BertPooler
import copy

class MyAveragePool(nn.Module):
    def __init__(self, dim):
        super(MyAveragePool, self).__init__()
        self.dim = dim

    def forward(self, inputs):
        result = inputs.mean(dim=self.dim)
        return result


class BERT_SPC(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_SPC, self).__init__()
        self.bert = bert
        self.opt = opt
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)
        self.sotfmax = nn.Softmax(dim=-1)

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
        concat_input_ids, concat_token_type_ids, mask_idx = inputs[0], inputs[1], inputs[2]
        bert_out = self.bert(input_ids=concat_input_ids, token_type_ids=concat_token_type_ids)[0]
        mask_out = self.dropout(self._getMask(bert_out, mask_idx))
        logits = self.dense(mask_out)
        return self.sotfmax(logits)


class BERT_ALTC(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_ALTC, self).__init__()
        self.opt = opt
        self.encode_bert_gloabl = copy.deepcopy(bert)

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, int(opt.bert_dim / 3), (k, opt.bert_dim)) for k in self.opt.kernel_size])
        self.maxs = nn.ModuleList(
            [nn.MaxPool1d(kernel_size=opt.max_seq_len - k + 1) for k in self.opt.kernel_size])

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(opt.dropout)
        self.pooling = MyAveragePool(dim=1)
        self.mha = BertSelfAttention(bert.config)
        self.concat_double_linear = nn.Linear(opt.bert_dim * 2, opt.bert_dim)
        self.gate = nn.Sequential(
            nn.Linear(opt.bert_dim * 2, opt.bert_dim * 2),
            nn.Sigmoid()
        )
        self.classifier_2 = nn.Linear(opt.bert_dim * 2, opt.polarities_dim)
        self.classifier_3 = nn.Linear(opt.bert_dim * 3, opt.polarities_dim)
        self.classifier_1 = nn.Linear(opt.bert_dim, opt.polarities_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        concat_input_ids, concat_token_type_ids, mask_idx = inputs[0], inputs[1], inputs[2]

        # embedding BERT编码
        bert_global_embedding = self.encode_bert_gloabl(
            input_ids=concat_input_ids, token_type_ids=concat_token_type_ids)[
            'last_hidden_state']

        # # pct全局语义细化
        # cnn_out1 = self.act(
        #     self.conv1(bert_global_embedding.unsqueeze(-1).transpose(1, 2)))  # b,n,d,1->b,d,n,1->b,x,n,1
        # cnn_out2 = self.act(self.conv2(cnn_out1)).transpose(1, 2).squeeze(-1)  # b,x,n,1->b,d,n,1->b,n,d,1
        # global_out = self.dropout(cnn_out2)
        #
        # # 语义融合 残差式 or 拼接式
        # if self.opt.residual:
        #     global_output = global_out + bert_global_embedding
        #     # global_output = nn.LayerNorm(global_out.shape).to(self.opt.device)(global_output)
        # else:
        #     concat_out = self.concat_double_linear(torch.cat([bert_global_embedding, global_out], -1))
        #     concat_out = self.mha(concat_out)[0]
        #     global_output = self.dropout(concat_out)

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

        # # 提取MASK所在位置
        # mask_out = self._getMask(global_output, mask_idx)
        # if self.opt.pool:
        #     pool_out = self.pooling(global_output)
        #     pool_mask_out = self.gate(torch.cat([mask_out, pool_out], -1))
        #     concat_out = torch.cat([pool_mask_out, local_out], -1)
        #     output = self.classifier_3(concat_out)
        # else:
        #     concat_out = torch.cat([mask_out, local_out], -1)
        output = self.classifier_1(local_out)
        return self.softmax(output)


class BERT_MPCG(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_MPCG, self).__init__()
        self.opt = opt
        self.encode_bert_gloabl = copy.deepcopy(bert)

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
        self.mha = BertSelfAttention(bert.config)
        self.concat_double_linear = nn.Linear(opt.bert_dim * 2, opt.bert_dim)
        self.gate = nn.Sequential(
            nn.Linear(opt.bert_dim * 2, opt.bert_dim * 2),
            nn.Sigmoid()
        )
        self.classifier_2 = nn.Linear(opt.bert_dim * 2, opt.polarities_dim)
        self.classifier_3 = nn.Linear(opt.bert_dim * 3, opt.polarities_dim)
        self.softmax = nn.Softmax(dim=-1)

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
        concat_input_ids, concat_token_type_ids, mask_idx = inputs[0], inputs[1], inputs[2]

        # embedding BERT编码
        bert_global_embedding = self.encode_bert_gloabl(
            input_ids=concat_input_ids, token_type_ids=concat_token_type_ids)[
            'last_hidden_state']

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

        # # 标准cnn提取局部语义
        # local_out = None
        # for conv, max in zip(self.convs, self.maxs):
        #     conv_input = bert_global_embedding.unsqueeze(1)  # (B,N,D)->(B,1,N,D)
        #     conv_output = self.act(conv(conv_input).squeeze(-1))  # (B,1,N,D)->(B,O_C,N-K+1)
        #     max_output = max(conv_output).squeeze(-1)  # (B,O_C,N-K+1)->(B,O_C)
        #     if local_out is None:
        #         local_out = max_output
        #     else:
        #         local_out = torch.cat([local_out, max_output], -1)

        # # 提取MASK所在位置
        # mask_out = self._getMask(global_output, mask_idx)
        # if self.opt.pool:
        #     pool_out = self.pooling(global_output)
        #     pool_mask_out = self.gate(torch.cat([mask_out, pool_out], -1))
        #     concat_out = torch.cat([pool_mask_out, local_out], -1)
        #     output = self.classifier_3(concat_out)
        # else:
        #     concat_out = torch.cat([mask_out, local_out], -1)
        #     output = self.classifier_2(concat_out)

        mask_out = self._getMask(global_output, mask_idx)
        pool_out = self.pooling(global_output)
        pool_mask_out = self.gate(torch.cat([mask_out, pool_out], -1))
        output = self.classifier_2(pool_mask_out)
        return self.softmax(output)


class BERT_AB_CSR(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_AB_CSR, self).__init__()
        self.opt = opt
        self.encode_bert_gloabl = copy.deepcopy(bert)
        self.classifier_1 = nn.Linear(opt.bert_dim, opt.polarities_dim)
        self.softmax = nn.Softmax(dim=-1)

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
        concat_input_ids, concat_token_type_ids, mask_idx = inputs[0], inputs[1], inputs[2]

        # embedding BERT编码
        bert_global_embedding = self.encode_bert_gloabl(
            input_ids=concat_input_ids, token_type_ids=concat_token_type_ids)[
            'last_hidden_state']

        # 提取MASK所在位置
        mask_out = self._getMask(bert_global_embedding, mask_idx)
        output = self.classifier_1(mask_out)
        return self.softmax(output)


class BERT_AB_PROMPT(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_AB_PROMPT, self).__init__()
        self.opt = opt
        self.encode_bert_gloabl = copy.deepcopy(bert)

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
        self.pooler = BertPooler(bert.config)
        self.mha = BertSelfAttention(bert.config)
        self.concat_double_linear = nn.Linear(opt.bert_dim * 2, opt.bert_dim)
        self.gate = nn.Sequential(
            nn.Linear(opt.bert_dim * 2, opt.bert_dim * 2),
            nn.Sigmoid()
        )
        self.classifier_2 = nn.Linear(opt.bert_dim * 2, opt.polarities_dim)
        self.classifier_3 = nn.Linear(opt.bert_dim * 3, opt.polarities_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        concat_input_ids, concat_token_type_ids = inputs[0], inputs[1]
        # embedding BERT编码
        bert_global_embedding = self.encode_bert_gloabl(
            input_ids=concat_input_ids, token_type_ids=concat_token_type_ids)[
            'last_hidden_state']

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
        return self.softmax(output)
