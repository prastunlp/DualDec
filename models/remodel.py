
import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import pdb
from utils import torch_utils, loader
from models import layers, entitymodel
from transformers import BertModel

model_name = 'bert-base-cased'


class REModel(object):

    def __init__(self, opt, W_entityclass_emb=None, W_relationclass_emb=None):
        self.opt = opt
        self.model = RelationModel(opt, W_entityclass_emb, W_relationclass_emb)
        self.relation_criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.entity_criterion = nn.CrossEntropyLoss(reduction='none')
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.relation_criterion.cuda()
            self.entity_criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'], opt['weight_decay'])

    def update(self, batch):
        # (T, STP , Rel, K1, K2, S1, S2, O1, O2, TS1, TS2, TO1, TO2, Nearest_S1, Distance_S1, Distance_to_subj, Nearest_O1, Distance_O1, T_Rel, orig_idx)
        inputs = [Variable(torch.LongTensor(b).cuda()) for b in batch[0:5]]   #(T, STP, Rel, K1, K2)
        subj_start_binary = Variable(torch.LongTensor(batch[5]).cuda()).float()
        subj_end_binary = Variable(torch.LongTensor(batch[6]).cuda()).float()
        obj_start_binary = Variable(torch.LongTensor(batch[7]).cuda())
        obj_end_binary = Variable(torch.LongTensor(batch[8]).cuda())
        subj_start_type = Variable(torch.LongTensor(batch[9]).cuda())
        subj_end_type = Variable(torch.LongTensor(batch[10]).cuda())
        obj_start_type = Variable(torch.LongTensor(batch[11]).cuda())
        obj_end_type = Variable(torch.LongTensor(batch[12]).cuda())
        #nearest_subj_start_position_for_each_token = Variable(torch.LongTensor(batch[13]).cuda())
        distance_to_nearest_subj_start = Variable(torch.LongTensor(batch[14]).cuda())
        distance_to_subj = Variable(torch.LongTensor(batch[15]).cuda())
        #nearest_obj_start_position_for_each_token = Variable(torch.LongTensor(batch[16]).cuda())
        distance_to_nearest_obj_start = Variable(torch.LongTensor(batch[17]).cuda())
        relation_binary = Variable(torch.FloatTensor(batch[18]).cuda())
        # if(relation_binary.size() == self.opt['num_class']):
        #     relation_binary = relation_binary.unsqueeze(0)
        mask = (inputs[0].data > 0).float()
        batch_size, rel_nums = relation_binary.size()

        self.model.train()
        self.optimizer.zero_grad()

        relation_logits, subj_start_logits, subj_end_logits, obj_start_logits, obj_end_logits = self.model(inputs, mask,
                                                                                          distance_to_nearest_subj_start,
                                                                                          distance_to_subj,
                                                                                          distance_to_nearest_obj_start)



        subj_start_loss = self.entity_criterion(subj_start_logits.view(-1, self.opt['num_subj_type'] + 1),
                                             subj_start_type.view(-1).squeeze()).view_as(mask)
        subj_start_loss = torch.sum(subj_start_loss.mul(mask.float())) / torch.sum(mask.float())

        subj_end_loss = self.entity_criterion(subj_end_logits.view(-1, self.opt['num_subj_type'] + 1),
                                           subj_end_type.view(-1).squeeze()).view_as(mask)
        subj_end_loss = torch.sum(subj_end_loss.mul(mask.float())) / torch.sum(mask.float())

        obj_start_loss = self.entity_criterion(obj_start_logits.view(-1, self.opt['num_subj_type'] + 1),
                                            obj_start_type.view(-1).squeeze()).view_as(mask)
        obj_start_loss = torch.sum(obj_start_loss.mul(mask.float())) / torch.sum(mask.float())

        obj_end_loss = self.entity_criterion(obj_end_logits.view(-1, self.opt['num_subj_type'] + 1),
                                          obj_end_type.view(-1).squeeze()).view_as(mask)
        obj_end_loss = torch.sum(obj_end_loss.mul(mask.float())) / torch.sum(mask.float())


        relation_loss = self.relation_criterion(relation_logits.view(-1, self.opt['num_class']),
                                               relation_binary.view(-1, self.opt['num_class']))
        relation_loss = torch.sum(relation_loss)/(batch_size * rel_nums)


        loss = (subj_start_loss + subj_end_loss) + (obj_start_loss + obj_end_loss)  + relation_loss

        loss.backward()

        self.optimizer.step()
        loss_val = loss.data.item()
        return loss_val

    def predict_relation_per_instance(self, tokens):

        tokens = Variable(torch.LongTensor(tokens).cuda())
        tokens = tokens.unsqueeze(0)
        batch_size, seq_len = tokens.size()
        mask = (tokens.data > 0).float()

        self.model.eval()

        outputs = self.model.model(tokens, attention_mask=mask)
        seq_outputs = outputs[0]
        pool_outputs = outputs[1]

        rel_inputs = pool_outputs
        #rel_inputs = self.model.LN(rel_inputs)
        rel_inputs = self.model.dropout(rel_inputs)
        relation_logits = self.model.liner_relation(rel_inputs)
        rel_back = rel_inputs

        lst = [i for i in range(self.model.relationclass_num)]
        relationclass_ = Variable(torch.LongTensor(lst).cuda())
        relationclass_emb = self.model.relationclass_emb(relationclass_)
        relationclass_emb = relationclass_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        rel_outputs = relationclass_emb

        return relation_logits, seq_outputs, rel_outputs, mask

    def predict_subject_per_instance(self, seq_outputs, rel_outputs, rel_id, mask):

        rel_id = Variable(torch.LongTensor(rel_id).cuda())

        mask = mask.long()
        if mask.shape[0] == 1:
            seq_lens = [mask.sum(1).squeeze()]
        else:
            seq_lens = list(mask.sum(1).squeeze())

        self.model.eval()

        subj_start_logits, subj_start_outputs = self.model.subj_sublayer.predict_subj_start(seq_outputs, rel_outputs, rel_id, seq_lens)

        subj_start = np.argmax(subj_start_logits, 1)
        nearest_subj_position_for_each_token, distance_to_nearest_subj_start = loader.get_nearest_start_position([subj_start])
        nearest_subj_position_for_each_token, distance_to_nearest_subj_start = Variable(
                                         torch.LongTensor(np.array(nearest_subj_position_for_each_token)).cuda()), Variable(
                                         torch.LongTensor(np.array(distance_to_nearest_subj_start)).cuda())

        subj_end_logits = self.model.subj_sublayer.predict_subj_end(seq_outputs, rel_outputs, rel_id, distance_to_nearest_subj_start, subj_start_outputs, seq_lens)

        return subj_start_logits, subj_end_logits

    def predict_object_per_instance(self, seq_outputs, rel_outputs, stp, rel_id, subj_start_position, subj_end_position, distance_to_subj, mask):

        mask = mask.long()
        if mask.shape[0] == 1:
            seq_lens = [mask.sum(1).squeeze()]
        else:
            seq_lens = list(mask.sum(1).squeeze())

        stp = Variable(torch.LongTensor(stp).cuda())
        rel_id = Variable(torch.LongTensor(rel_id).cuda())
        subj_start_position = Variable(torch.LongTensor(subj_start_position).cuda())
        subj_end_position = Variable(torch.LongTensor(subj_end_position).cuda())
        distance_to_subj = Variable(torch.LongTensor(distance_to_subj).cuda())

        self.model.eval()

        obj_start_logits, rel_subj_info, obj_start_outputs = self.model.obj_sublayer.predict_obj_start(seq_outputs, rel_outputs, stp, rel_id, subj_start_position,
                                                             subj_end_position, distance_to_subj, seq_lens)
        obj_start = np.argmax(obj_start_logits, 1)
        nearest_obj_start_position_for_each_token, distance_to_nearest_obj_start = loader.get_nearest_start_position([obj_start])
        nearest_obj_start_position_for_each_token, distance_to_nearest_obj_start = Variable(
                                        torch.LongTensor(np.array(nearest_obj_start_position_for_each_token)).cuda()), Variable(
                                        torch.LongTensor(np.array(distance_to_nearest_obj_start)).cuda())

        obj_end_logits = self.model.obj_sublayer.predict_obj_end(seq_outputs, rel_subj_info, distance_to_nearest_obj_start, obj_start_outputs, seq_lens)

        return obj_start_logits, obj_end_logits

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    def save(self, filename, epoch):
        params = {
            'model': self.model.state_dict(),
            'config': self.opt,
            'epoch': epoch
        }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']


class RelationModel(nn.Module):
    def __init__(self, opt, W_entityclass_emb=None, W_relationclass_emb=None):
        super(RelationModel, self).__init__()
        self.model = BertModel.from_pretrained(model_name, output_hidden_states=True)
        for param in self.model.parameters():
            param.requires_grad = True
        self.dropout = nn.Dropout(0.4)
        self.relationclass_num = opt['relationclass_num']
        self.relationclass_emb = nn.Embedding(opt['relationclass_num'], opt['word_emb_dim'])

        self.liner_relation = nn.Linear(opt['tokens_emb_dim'], opt['relationclass_num'])

        self.subj_sublayer = entitymodel.SubjTypeModel(opt)
        self.obj_sublayer = entitymodel.ObjTypeModel(opt, W_entityclass_emb)
        self.opt = opt
        self.use_cuda = opt['cuda']
        self.W_relationclass_emb = W_relationclass_emb
        self.init_weights()

    def init_weights(self):
        if self.W_relationclass_emb is None:
            self.relationclass_emb.weight.data.uniform_(-1.0, 1.0)
        else:
            self.W_relationclass_emb = torch.from_numpy(self.W_relationclass_emb)
            self.relationclass_emb.weight.data.copy_(self.W_relationclass_emb)

        self.liner_relation.bias.data.fill_(0)
        init.xavier_uniform_(self.liner_relation.weight, gain=1)

    def zero_state(self, batch_size):
        state_shape = (2 * self.opt['num_layers'], batch_size, self.opt['hidden_dim'])
        h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
        if self.use_cuda:
            return h0.cuda(), c0.cuda()
        else:
            return h0, c0

    def forward(self, inputs, mask, distance_to_nearest_subj_start, distance_to_subj, distance_to_nearest_obj_start):

        tokens, stp, rel_id, subj_start_position, subj_end_position = inputs
        batch_size, seq_len = tokens.size()
        outputs = self.model(tokens, attention_mask=mask)
        seq_outputs = outputs[0]
        pool_outputs = outputs[1]
        seq_all_layer_outputs = outputs[2]

        rel_inputs = pool_outputs
        rel_inputs = self.dropout(rel_inputs)
        relation_logits = self.liner_relation(rel_inputs)


        lst = [i for i in range(self.relationclass_num)]
        relationclass_ = Variable(torch.LongTensor(lst).cuda())
        relation_emb = self.relationclass_emb(relationclass_)
        relation_emb = relation_emb.unsqueeze(0).repeat(batch_size, 1, 1)
        rel_outputs = relation_emb

        subj_start_logits, subj_end_logits = self.subj_sublayer(seq_outputs, rel_outputs, rel_id, mask, distance_to_nearest_subj_start)
        obj_start_logits, obj_end_logits = self.obj_sublayer(seq_outputs, rel_outputs, stp, rel_id, subj_start_position,
                                                             subj_end_position, mask, distance_to_subj,
                                                             distance_to_nearest_obj_start)

        return relation_logits, subj_start_logits, subj_end_logits, obj_start_logits, obj_end_logits



