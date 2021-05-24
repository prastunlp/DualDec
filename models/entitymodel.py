import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
from utils import torch_utils

from models.layers import *


class SubjTypeModel(nn.Module):
    def __init__(self, opt):
        super(SubjTypeModel, self).__init__()
        self.opt = opt
        self.input_size = self.opt['tokens_emb_dim'] + self.opt['word_emb_dim']
        self.dropout = nn.Dropout(opt['dropout'])
        self.LN_subj_start = nn.LayerNorm(self.opt['tokens_emb_dim'] + self.opt['word_emb_dim'],elementwise_affine=True)
        self.LN_subj_end = nn.LayerNorm(self.opt['tokens_emb_dim'] + self.opt['word_emb_dim'] +  opt['position_emb_dim'], elementwise_affine=True)

        self.distance_to_subj_start_embedding = nn.Embedding(500, opt['position_emb_dim'])
        self.rnn_start = nn.LSTM(self.input_size, 768 // 2, 1, batch_first=True, bidirectional=True)
        self.rnn_end = nn.LSTM(self.input_size + opt['position_emb_dim'], 768 // 2, 1, batch_first=True,bidirectional=True)

        self.linear_subj_start = nn.Linear(768, opt['num_subj_type'] + 1)
        self.linear_subj_end = nn.Linear(768, opt['num_subj_type'] + 1)
        self.init_weights()

    def zero_state(self, batch_size):
        state_shape = (2, batch_size, 768 // 2)
        h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
        return h0.cuda(), c0.cuda()

    def init_weights(self):

        self.distance_to_subj_start_embedding.weight.data.uniform_(-1.0, 1.0)

        self.linear_subj_start.bias.data.fill_(0)
        init.xavier_uniform_(self.linear_subj_start.weight, gain=1) # initialize linear layer

        self.linear_subj_end.bias.data.fill_(0)
        init.xavier_uniform_(self.linear_subj_end.weight, gain=1) # initialize linear layer

    def forward(self, seq_outputs, rel_outputs, rel_id, masks,  distance_to_nearest_subj_start):

        rel_input_size =  self.opt['word_emb_dim']
        batch_size, seq_len, seq_hidden_dim = seq_outputs.shape

        mask = masks.long()
        if mask.shape[0] == 1:
            seq_lens = [mask.sum(1).squeeze()]
        else:
            seq_lens = list(mask.sum(1).squeeze())

        rel_hidden = torch.gather(rel_outputs, dim=1, index=rel_id.unsqueeze(2).repeat(1, 1, rel_input_size)).squeeze(1)

        subj_start_inputs = torch.cat([seq_outputs, seq_and_vec(seq_len,rel_hidden)], dim=2)
        subj_start_inputs = self.LN_subj_start(subj_start_inputs)
        subj_start_inputs = self.dropout(subj_start_inputs)

        h0, c0 = self.zero_state(batch_size)
        subj_start_outputs = nn.utils.rnn.pack_padded_sequence(subj_start_inputs, seq_lens, batch_first=True)
        subj_start_outputs, (ht, ct) = self.rnn_start(subj_start_outputs, (h0, c0))
        subj_start_outputs, output_lens = nn.utils.rnn.pad_packed_sequence(subj_start_outputs, batch_first=True)


        distance_to_subj_start_emb = self.distance_to_subj_start_embedding(distance_to_nearest_subj_start)
        subj_end_inputs = torch.cat([subj_start_outputs, distance_to_subj_start_emb, seq_and_vec(seq_len, rel_hidden)], dim=2)
        subj_end_inputs = self.LN_subj_end(subj_end_inputs)
        subj_end_inputs = self.dropout(subj_end_inputs)

        subj_end_outputs = nn.utils.rnn.pack_padded_sequence(subj_end_inputs, seq_lens, batch_first=True)
        subj_end_outputs, (ht, ct) = self.rnn_end(subj_end_outputs, (h0, c0))
        subj_end_outputs, output_lens = nn.utils.rnn.pad_packed_sequence(subj_end_outputs, batch_first=True)


        subj_start_outputs = F.dropout(subj_start_outputs, p=0.4, training=True)
        subj_start_logits = self.linear_subj_start(subj_start_outputs)
        subj_end_outputs = F.dropout(subj_end_outputs, p=0.4, training=True)
        subj_end_logits = self.linear_subj_end(subj_end_outputs)

        return subj_start_logits, subj_end_logits

    def predict_subj_start(self, seq_outputs, rel_outputs, rel_id, seq_lens):

        rel_input_size = self.opt['word_emb_dim']
        batch_size, seq_len, seq_hidden_dim = seq_outputs.shape
        rel_hidden = torch.gather(rel_outputs, dim=1, index=rel_id.unsqueeze(2).repeat(1, 1, rel_input_size)).squeeze(1)


        subj_start_inputs = torch.cat([seq_outputs, seq_and_vec(seq_len, rel_hidden)], dim=2)
        subj_start_inputs = self.LN_subj_start(subj_start_inputs)
        subj_start_inputs = self.dropout(subj_start_inputs)

        h0, c0 = self.zero_state(batch_size)
        subj_start_outputs = nn.utils.rnn.pack_padded_sequence(subj_start_inputs, seq_lens, batch_first=True)
        subj_start_outputs, (ht, ct) = self.rnn_start(subj_start_outputs, (h0, c0))
        subj_start_outputs, output_lens = nn.utils.rnn.pad_packed_sequence(subj_start_outputs, batch_first=True)
        subj_start_outputs_back = subj_start_outputs

        subj_start_outputs = F.dropout(subj_start_outputs, p=0.4, training=False)
        subj_start_logits = self.linear_subj_start(subj_start_outputs)

        return subj_start_logits.squeeze(-1)[0].data.cpu().numpy(), subj_start_outputs_back

    def predict_subj_end(self, seq_outputs, rel_outputs, rel_id, distance_to_nearest_subj_start, subj_start_outputs, seq_lens):

        rel_input_size = self.opt['word_emb_dim']
        rel_hidden = torch.gather(rel_outputs, dim=1, index=rel_id.unsqueeze(2).repeat(1, 1, rel_input_size)).squeeze(1)
        batch_size, seq_len, seq_hidden_dim = seq_outputs.shape


        distance_to_subj_start_emb = self.distance_to_subj_start_embedding(distance_to_nearest_subj_start)
        subj_end_inputs = torch.cat([subj_start_outputs, distance_to_subj_start_emb, seq_and_vec(seq_len, rel_hidden)], dim=2)
        subj_end_inputs = self.LN_subj_end(subj_end_inputs)
        subj_end_inputs = self.dropout(subj_end_inputs)

        h0, c0 = self.zero_state(batch_size)
        subj_end_outputs = nn.utils.rnn.pack_padded_sequence(subj_end_inputs, seq_lens, batch_first=True)
        subj_end_outputs, (ht, ct) = self.rnn_end(subj_end_outputs, (h0, c0))
        subj_end_outputs, output_lens = nn.utils.rnn.pad_packed_sequence(subj_end_outputs, batch_first=True)

        subj_end_outputs = F.dropout(subj_end_outputs, p=0.4, training=False)
        subj_end_logits = self.linear_subj_end(subj_end_outputs)

        return subj_end_logits.squeeze(-1)[0].data.cpu().numpy()

class ObjTypeModel(nn.Module):

    def __init__(self, opt, W_entityclass_emb):
        super(ObjTypeModel, self).__init__()
        self.opt = opt
        self.input_size = 2*self.opt['tokens_emb_dim'] + self.opt['word_emb_dim']
        self.entityclass_num = opt['entityclass_num']
        self.entityclass_emb = nn.Embedding(opt['entityclass_num'], opt['word_emb_dim'])

        self.dropout = nn.Dropout(opt['dropout'])

        self.distance_to_subj_embedding = nn.Embedding(1200, opt['position_emb_dim'])
        self.distance_to_obj_start_embedding = nn.Embedding(500, opt['position_emb_dim'])

        self.LN_obj_start = nn.LayerNorm(2*self.opt['tokens_emb_dim'] + self.opt['word_emb_dim'] + opt['position_emb_dim'],
                                          elementwise_affine=True)
        self.LN_obj_end = nn.LayerNorm(2*self.opt['tokens_emb_dim'] + self.opt['word_emb_dim'] + 2*opt['position_emb_dim'],
                                          elementwise_affine=True)

        self.rnn_start = nn.LSTM(self.input_size + opt['position_emb_dim'], 768 // 2, 1, batch_first=True,
                                 bidirectional=True)
        self.rnn_end = nn.LSTM(self.input_size + 2 * opt['position_emb_dim'], 768 // 2, 1, batch_first=True,
                               bidirectional=True)

        self.linear_obj_start = nn.Linear(768,opt['num_subj_type'] + 1)

        self.linear_obj_end = nn.Linear(768,opt['num_subj_type'] + 1)

        self.W_entityclass_emb = W_entityclass_emb
        self.init_weights()

    def zero_state(self, batch_size):
        state_shape = (2, batch_size, 768 // 2)
        h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
        return h0.cuda(), c0.cuda()

    def init_weights(self):

        if self.W_entityclass_emb is None:
            self.entityclass_emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        else:
            self.W_entityclass_emb = torch.from_numpy(self.W_entityclass_emb)
            self.entityclass_emb.weight.data.copy_(self.W_entityclass_emb)

        self.distance_to_subj_embedding.weight.data.uniform_(-1.0, 1.0)
        self.distance_to_obj_start_embedding.weight.data.uniform_(-1.0, 1.0)

        self.linear_obj_start.bias.data.fill_(0)
        init.xavier_uniform_(self.linear_obj_start.weight, gain=1)
        self.linear_obj_end.bias.data.fill_(0)
        init.xavier_uniform_(self.linear_obj_end.weight, gain=1)


    def forward(self, seq_outputs, rel_outputs, stp, rel_id, subj_start_position, subj_end_position, masks, distance_to_subj, distance_to_nearest_obj_start):

        rel_input_size = self.opt['word_emb_dim']
        ent_input_size = self.opt['word_emb_dim']
        batch_size, seq_len, seq_hidden_dim = seq_outputs.shape

        mask = masks.long()
        if mask.shape[0] == 1:
            seq_lens = [mask.sum(1).squeeze()]
        else:
            seq_lens = list(mask.sum(1).squeeze())

        rel_hidden = torch.gather(rel_outputs, dim=1, index=rel_id.unsqueeze(2).repeat(1, 1, rel_input_size)).squeeze(1)

        subject_start_hidden = torch.gather(seq_outputs, dim=1, index=subj_start_position.unsqueeze(2).repeat(1, 1, seq_hidden_dim)).squeeze(1)
        subject_end_hidden = torch.gather(seq_outputs, dim=1, index=subj_end_position.unsqueeze(2).repeat(1, 1, seq_hidden_dim)).squeeze(1)
        subj_info = torch.stack([subject_start_hidden,subject_end_hidden])
        subj_info = torch.sum(subj_info, dim=0)

        lst = [i for i in range(self.entityclass_num)]
        entityclass_ = Variable(torch.LongTensor(lst).cuda())
        entityclass_emb = self.entityclass_emb(entityclass_)
        entityclass_emb = entityclass_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        subject_type = torch.gather(entityclass_emb, dim=1, index=stp.unsqueeze(2).repeat(1, 1, ent_input_size)).squeeze(1)

        label_emb = torch.stack([rel_hidden,subject_type])
        label_emb = torch.sum(label_emb, dim=0)

        distance_to_subj_emb = self.distance_to_subj_embedding(distance_to_subj+600)

        rel_subj_info = torch.cat([seq_and_vec(seq_len, subj_info), seq_and_vec(seq_len, label_emb)], dim=2)
        obj_start_inputs = torch.cat([seq_outputs, rel_subj_info , distance_to_subj_emb], dim=2)
        obj_start_inputs = self.LN_obj_start(obj_start_inputs)
        obj_start_inputs = self.dropout(obj_start_inputs)

        h0, c0 = self.zero_state(batch_size)
        obj_start_outputs = nn.utils.rnn.pack_padded_sequence(obj_start_inputs, seq_lens, batch_first=True)
        obj_start_outputs, (ht, ct) = self.rnn_start(obj_start_outputs, (h0, c0))
        obj_start_outputs, output_lens = nn.utils.rnn.pad_packed_sequence(obj_start_outputs, batch_first=True)

        distance_to_nearest_obj_start_emb = self.distance_to_obj_start_embedding(distance_to_nearest_obj_start)
        obj_end_inputs = torch.cat([obj_start_outputs, rel_subj_info, distance_to_subj_emb, distance_to_nearest_obj_start_emb], dim=2)
        obj_end_inputs = self.LN_obj_end(obj_end_inputs)
        obj_end_inputs = self.dropout(obj_end_inputs)
        obj_end_outputs = nn.utils.rnn.pack_padded_sequence(obj_end_inputs, seq_lens, batch_first=True)
        obj_end_outputs, (ht, ct) = self.rnn_end(obj_end_outputs, (h0, c0))
        obj_end_outputs, output_lens = nn.utils.rnn.pad_packed_sequence(obj_end_outputs, batch_first=True)

        obj_start_outputs = F.dropout(obj_start_outputs, p=0.4, training=True)
        obj_start_logits = self.linear_obj_start(obj_start_outputs)

        obj_end_outputs = F.dropout(obj_end_outputs, p=0.4, training=True)
        obj_end_logits = self.linear_obj_end(obj_end_outputs)

        return obj_start_logits, obj_end_logits

    def predict_obj_start(self, seq_outputs, rel_outputs, stp, rel_id, subj_start_position, subj_end_position, distance_to_subj, seq_lens):

        rel_input_size = self.opt['word_emb_dim']
        ent_input_size = self.opt['word_emb_dim']
        batch_size, seq_len, seq_hidden_dim = seq_outputs.shape
        rel_hidden = torch.gather(rel_outputs, dim=1, index=rel_id.unsqueeze(2).repeat(1, 1, rel_input_size)).squeeze(1)

        subject_start_hidden = torch.gather(seq_outputs, dim=1, index=subj_start_position.unsqueeze(2).repeat(1, 1,seq_hidden_dim)).squeeze(1)

        subject_end_hidden = torch.gather(seq_outputs, dim=1,index=subj_end_position.unsqueeze(2).repeat(1, 1, seq_hidden_dim)).squeeze(1)
        subj_info = torch.stack([subject_start_hidden, subject_end_hidden])
        subj_info = torch.mean(subj_info, dim=0)

        lst = [i for i in range(self.entityclass_num)]
        entityclass_ = Variable(torch.LongTensor(lst).cuda())
        entityclass_emb = self.entityclass_emb(entityclass_)
        entityclass_emb = entityclass_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        subject_type = torch.gather(entityclass_emb, dim=1,
                                    index=stp.unsqueeze(2).repeat(1, 1, ent_input_size)).squeeze(1)

        label_emb = torch.stack([rel_hidden, subject_type])
        label_emb = torch.sum(label_emb, dim=0)

        distance_to_subj_emb = self.distance_to_subj_embedding(distance_to_subj + 600)

        rel_subj_info = torch.cat([seq_and_vec(seq_len, subj_info), seq_and_vec(seq_len, label_emb), distance_to_subj_emb], dim=2)

        obj_start_inputs = torch.cat([seq_outputs, rel_subj_info], dim=2)
        obj_start_inputs = self.LN_obj_start(obj_start_inputs)
        obj_start_inputs = self.dropout(obj_start_inputs)

        h0, c0 = self.zero_state(batch_size)
        obj_start_outputs = nn.utils.rnn.pack_padded_sequence(obj_start_inputs, seq_lens, batch_first=True)
        obj_start_outputs, (ht, ct) = self.rnn_start(obj_start_outputs, (h0, c0))
        obj_start_outputs, output_lens = nn.utils.rnn.pad_packed_sequence(obj_start_outputs, batch_first=True)
        obj_start_outputs_back = obj_start_outputs

        obj_start_outputs = F.dropout(obj_start_outputs, p=0.4, training=False)
        obj_start_logits = self.linear_obj_start(obj_start_outputs)

        return obj_start_logits.squeeze(-1)[0].data.cpu().numpy(), rel_subj_info, obj_start_outputs_back


    def predict_obj_end(self, seq_outputs, rel_subj_info, distance_to_nearest_obj_start, obj_start_outputs, seq_lens):

        distance_to_nearest_obj_start_emb = self.distance_to_obj_start_embedding(distance_to_nearest_obj_start)
        batch_size, seq_len, seq_hidden_dim = seq_outputs.shape
        obj_end_inputs = torch.cat([obj_start_outputs, rel_subj_info, distance_to_nearest_obj_start_emb], dim=2)
        obj_end_inputs = self.LN_obj_end(obj_end_inputs)
        obj_end_inputs = self.dropout(obj_end_inputs)
        h0, c0 = self.zero_state(batch_size)
        obj_end_outputs = nn.utils.rnn.pack_padded_sequence(obj_end_inputs, seq_lens, batch_first=True)
        obj_end_outputs, (ht, ct) = self.rnn_end(obj_end_outputs, (h0, c0))
        obj_end_outputs, output_lens = nn.utils.rnn.pad_packed_sequence(obj_end_outputs, batch_first=True)

        obj_end_outputs = F.dropout(obj_end_outputs, p=0.4, training=False)
        obj_end_logits = self.linear_obj_end(obj_end_outputs)

        return obj_end_logits.squeeze(-1)[0].data.cpu().numpy()


