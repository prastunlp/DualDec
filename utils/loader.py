import numpy as np
import random
from random import choice
import collections


global num

def get_nearest_start_position(S1):
    nearest_start_list = []
    current_distance_list = []
    for start_pos_list in S1:
        nearest_start_pos = []
        current_start_pos = 0
        current_pos = []
        flag = False
        for i, start_label in enumerate(start_pos_list):
            if start_label > 0:
                current_start_pos = i
                flag = True
            nearest_start_pos.append(current_start_pos)
            if flag > 0:
                if i-current_start_pos > 10:
                    current_pos.append(499)
                else:
                    current_pos.append(i-current_start_pos)
            else:
                current_pos.append(499)
        nearest_start_list.append(nearest_start_pos)
        current_distance_list.append(current_pos)
    return nearest_start_list, current_distance_list

def seq_padding(X):
    L = [len(x) for x in X]
    ML = max(L)
    return [x + [0] * (ML - len(x)) for x in X]

def get_positions(start_idx, end_idx, length):
    """ Get subj/obj position sequence. """
    return list(range(-start_idx, 0)) + [0]*(end_idx - start_idx + 1) + \
            list(range(1, length-end_idx))


def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]

def getTokenizer_forseq(tokenizer, lstseq):
    tokens = ['[CLS]']
    for token in lstseq:
        token = tokenizer.tokenize(token)
        for t in token:
            tokens.append(t)
        tokens.append('[unused1]')
    tokens.append('[SEP]')
    return tokens

def getTokenizer_forwords(tokenizer, words):
    words = words.split(' ')
    tokens = []
    for token in words:
        token = tokenizer.tokenize(token)
        for t in token:
            tokens.append(t)
        tokens.append('[unused1]')
    return tokens

def find_head_idx(text, entity):
    entity_len = len(entity)
    for i in range(len(text)):
        if text[i: i + entity_len] == entity:
            return i
    return -1

class DataLoader(object):
    def __init__(self, tokenizer, data, predicate2id, subj_type2id, obj_type2id, batch_size=64, evaluation=False, dataflag=0):

        self.batch_size = batch_size
        
        self.predicate2id = predicate2id
        self.subj_type2id = subj_type2id
        self.obj_type2id = obj_type2id
        if dataflag!=0:
            data = self.other_preprocess_bert(data, tokenizer)
        else:
            data = self.preprocess_bert(data,tokenizer)
        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        self.num_examples = len(data)
        self.data = [data[i:i+batch_size] for i in range(0, len(data), batch_size)]

    def other_preprocess_bert(self, data, tokenizer):
        processed = []
        for d in data:
            text = d['sentText']
            text = text.split(' ')
            if len(text) > 100:
                continue
            tokens = getTokenizer_forseq(tokenizer, text)  # tokenize text
            if len(tokens) > 512:
                tokens = tokens[:512]
            tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
            rels_dict = {}
            rel_nums = len(self.predicate2id)
            tokens_rel = [0] * rel_nums
            for triple in d['relationMentions']:
                triple = (getTokenizer_forwords(tokenizer, triple['em1Text']), triple['label'], getTokenizer_forwords(tokenizer, triple['em2Text']))
                sub_start_idx = find_head_idx(tokens, triple[0])
                obj_start_idx = find_head_idx(tokens, triple[2])
                if sub_start_idx == -1 or obj_start_idx == -1:
                    continue
                sub_end_idx = sub_start_idx+len(triple[0])-1
                obj_end_idx = obj_start_idx+len(triple[2])-1
                sub_type = 1
                obj_type = 1
                rel_id =  self.predicate2id[triple[1]]
                tokens_rel[rel_id] = 1
                if rel_id not in rels_dict:
                    rels_dict[rel_id] = []
                rels_dict[rel_id].append((sub_start_idx,sub_end_idx,sub_type,obj_start_idx,obj_end_idx,obj_type))
            if rels_dict:
                rel_ids = list(rels_dict.keys())
                for rel_id in rel_ids:
                    rel_triples = rels_dict[rel_id]
                    items = {}
                    subj_type = collections.defaultdict(list)
                    obj_type = collections.defaultdict(list)
                    for triple in rel_triples:
                        subj_key = (triple[0],triple[1])
                        subj_type[subj_key].append(triple[2])
                        if subj_key not in items:
                            items[subj_key] = []
                        items[subj_key].append((triple[3],triple[4]))
                        obj_type[(triple[3],triple[4])].append(triple[5])

                    if items:
                        s1, s2 = [0] * len(tokens), [0] * len(tokens)
                        ts1, ts2 = [0] * len(tokens), [0] * len(tokens)

                        for j in items:
                            s1[j[0]] = 1
                            s2[j[1]] = 1
                            stp = choice(subj_type[j])
                            ts1[j[0]] = stp
                            ts2[j[1]] = stp
                        k1, k2 = choice(list(items.keys()))
                        o1, o2 = [0] * len(tokens), [0] * len(tokens)
                        to1, to2 = [0] * len(tokens), [0] * len(tokens)
                        distance_to_subj = get_positions(k1, k2, len(tokens))

                        for j in items[(k1, k2)]:
                            o1[j[0]] = 1
                            o2[j[1]] = 1
                            otp = choice(obj_type[(j[0], j[1])])
                            to1[j[0]] = otp
                            to2[j[1]] = otp
                        processed += [(tokens_ids, tokens_rel, [k1], [k2], s1, s2, o1, o2, ts1, ts2, to1, to2,[rel_id],
                                       distance_to_subj, [stp])]
        return processed

    def preprocess_bert(self, data, tokenizer):
        processed = []
        for d in data:
            text = d['tokens']
            if len(text) > 150:
                continue
            tokens = getTokenizer_forseq(tokenizer, text)
            if len(tokens) > 512:
                tokens = tokens[:512]
            tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
            rels_dict = {}
            rel_nums = len(self.predicate2id)
            tokens_rel = [0] * rel_nums
            for triple,triple_details in zip(d['spo_list'],d['spo_details']):
                triple = (getTokenizer_forwords(tokenizer, triple[0]), triple[1], getTokenizer_forwords(tokenizer, triple[2]))
                sub_start_idx = find_head_idx(tokens, triple[0])
                obj_start_idx = find_head_idx(tokens, triple[2])
                if sub_start_idx == -1 or obj_start_idx == -1:
                    continue
                sub_end_idx = sub_start_idx+len(triple[0])-1
                obj_end_idx = obj_start_idx+len(triple[2])-1
                sub_type = self.subj_type2id[triple_details[2]]
                obj_type = self.obj_type2id[triple_details[6]]
                rel_id =  self.predicate2id[triple[1]] - 1
                tokens_rel[rel_id] = 1
                if rel_id not in rels_dict:
                    rels_dict[rel_id] = []
                rels_dict[rel_id].append((sub_start_idx,sub_end_idx,sub_type,obj_start_idx,obj_end_idx,obj_type))
            if rels_dict:
                rel_ids = list(rels_dict.keys())
                for rel_id in rel_ids:
                    rel_triples = rels_dict[rel_id]
                    items = {}
                    subj_type = collections.defaultdict(list)
                    obj_type = collections.defaultdict(list)
                    for triple in rel_triples:
                        subj_key = (triple[0],triple[1])
                        subj_type[subj_key].append(triple[2])
                        if subj_key not in items:
                            items[subj_key] = []
                        items[subj_key].append((triple[3],triple[4]))
                        obj_type[(triple[3],triple[4])].append(triple[5])

                    if items:
                        s1, s2 = [0] * len(tokens), [0] * len(tokens)
                        ts1, ts2 = [0] * len(tokens), [0] * len(tokens)

                        for j in items:
                            s1[j[0]] = 1
                            s2[j[1]] = 1
                            stp = choice(subj_type[j])
                            ts1[j[0]] = stp
                            ts2[j[1]] = stp
                        k1, k2 = choice(list(items.keys()))
                        o1, o2 = [0] * len(tokens), [0] * len(tokens)
                        to1, to2 = [0] * len(tokens), [0] * len(tokens)
                        distance_to_subj = get_positions(k1, k2, len(tokens))

                        for j in items[(k1, k2)]:
                            o1[j[0]] = 1
                            o2[j[1]] = 1
                            otp = choice(obj_type[(j[0], j[1])])
                            to1[j[0]] = otp
                            to2[j[1]] = otp
                        processed += [(tokens_ids, tokens_rel, [k1], [k2], s1, s2, o1, o2, ts1, ts2, to1, to2,[rel_id],
                                       distance_to_subj, [stp])]
        return processed

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __getitem__(self, key):
        """ Get a batch with index. """
        if not isinstance(key, int):
            raise TypeError
        if key < 0 or key >= len(self.data):
            raise IndexError
        batch = self.data[key]
        batch_size = len(batch)
        batch = list(zip(*batch))
        assert len(batch) == 15
        lens = [len(x) for x in batch[0]]

        batch, orig_idx = sort_all(batch, lens)
        T = np.array(seq_padding(batch[0]))
        T_Rel = np.array(batch[1])
        K1, K2 = np.array(batch[2]), np.array(batch[3])
        
        
        S1 = np.array(seq_padding(batch[4]))
        S2 = np.array(seq_padding(batch[5]))
        O1 = np.array(seq_padding(batch[6]))
        O2 = np.array(seq_padding(batch[7]))
        TS1 = np.array(seq_padding(batch[8]))
        TS2 = np.array(seq_padding(batch[9]))
        TO1 = np.array(seq_padding(batch[10]))
        TO2 = np.array(seq_padding(batch[11]))
        Rel = np.array(batch[12])
        STP = np.array(batch[14])
        Distance_to_subj = np.array(seq_padding(batch[13]))
        Nearest_S1, Distance_S1 = get_nearest_start_position(batch[4])
        Nearest_S1, Distance_S1 = np.array(seq_padding(Nearest_S1)), np.array(seq_padding(Distance_S1))
        Nearest_O1, Distance_O1 = get_nearest_start_position(batch[6])
        Nearest_O1, Distance_O1 = np.array(seq_padding(Nearest_O1)), np.array(seq_padding(Distance_O1))


        return (T, STP, Rel, K1, K2, S1, S2, O1, O2, TS1, TS2, TO1, TO2, Nearest_S1, Distance_S1, Distance_to_subj, Nearest_O1, Distance_O1, T_Rel, orig_idx)



