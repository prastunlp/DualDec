import numpy as np
from tqdm import tqdm
from utils import loader
import torch
import json
from transformers import BertTokenizer

def extract_items_bert(spo_list, tokenizer ,tokens_in, id2predicate, model):

    tokens = loader.getTokenizer_forseq(tokenizer, tokens_in)  # tokenize text
    if len(tokens) > 512:
        tokens = tokens[:512]
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    R = []
    relation_logits, seq_outputs, rel_outputs, mask = model.predict_relation_per_instance(tokens_ids)
    relation_logits = torch.sigmoid(relation_logits).data.cpu().numpy()
    rel_ids = np.where(relation_logits > 0.5)[1]
    for rel_id in rel_ids:
        _rel_id = np.array([[rel_id]])
        subj_start_logits, subj_end_logits = model.predict_subject_per_instance(seq_outputs, rel_outputs, _rel_id, mask)
        _s1, _s2 = np.argmax(subj_start_logits, 1), np.argmax(subj_end_logits, 1)
        for i, _ss1 in enumerate(_s1):
            if _ss1 > 0:
                _subject = ''
                for j, _ss2 in enumerate(_s2[i:]):
                    if _ss2 == _ss1:
                        subj = tokens[i: i + j + 1]
                        _subject = ''.join([hi.lstrip("##") for hi in subj])
                        _subject = ' '.join(_subject.split('[unused1]'))
                        _subject = _subject[:len(_subject) - 1]
                        break
                if _subject:
                    _k1, _k2 = np.array([[i]]), np.array([[i + j]])
                    stp = np.array([[_ss1]])
                    distance_to_subj = np.array([loader.get_positions(i, i + j, len(tokens))])
                    _o1, _o2 = model.predict_object_per_instance(seq_outputs, rel_outputs, stp, _rel_id, _k1, _k2, distance_to_subj, mask)
                    _o1, _o2 = np.argmax(_o1, 1), np.argmax(_o2, 1)
                    for i, _oo1 in enumerate(_o1):
                        if _oo1 > 0:
                            for j, _oo2 in enumerate(_o2[i:]):
                                if _oo2 == _oo1:
                                    obj = tokens[i: i + j + 1]
                                    _object = ''.join([oi.lstrip("##") for oi in obj])
                                    _object = ' '.join(_object.split('[unused1]'))
                                    _object = _object[:len(_object) - 1]
                                    _predicate = id2predicate[rel_id+1]
                                    R.append((_subject, _predicate, _object))
                                    break
    return set(R)

def other_extract_items_bert(tokenizer ,tokens_in, id2predicate, model):

    tokens_in = tokens_in.split(' ')
    tokens = loader.getTokenizer_forseq(tokenizer, tokens_in)  # tokenize text
    if len(tokens) > 512:
        tokens = tokens[:512]
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
    R = []
    relation_logits, seq_outputs, rel_outputs, mask = model.predict_relation_per_instance(tokens_ids)
    relation_logits = torch.sigmoid(relation_logits).data.cpu().numpy()
    rel_ids = np.where(relation_logits > 0.5)[1]
    for rel_id in rel_ids:
        _rel_id = np.array([[rel_id]])
        subj_start_logits, subj_end_logits = model.predict_subject_per_instance(seq_outputs, rel_outputs, _rel_id, mask)
        _s1, _s2 = np.argmax(subj_start_logits, 1), np.argmax(subj_end_logits, 1)
        for i, _ss1 in enumerate(_s1):
            if _ss1 > 0:
                _subject = ''
                for j, _ss2 in enumerate(_s2[i:]):
                    if _ss2 == _ss1:
                        subj = tokens[i: i + j + 1]
                        _subject = ''.join([hi.lstrip("##") for hi in subj])
                        _subject = ' '.join(_subject.split('[unused1]'))
                        _subject = _subject[:len(_subject) - 1]
                        break
                if _subject:
                    _k1, _k2 = np.array([[i]]), np.array([[i + j]])
                    stp = np.array([[_ss1]])
                    distance_to_subj = np.array([loader.get_positions(i, i + j, len(tokens))])
                    _o1, _o2 = model.predict_object_per_instance(seq_outputs, rel_outputs, stp, _rel_id, _k1, _k2, distance_to_subj, mask)
                    _o1, _o2 = np.argmax(_o1, 1), np.argmax(_o2, 1)
                    for i, _oo1 in enumerate(_o1):
                        if _oo1 > 0:
                            for j, _oo2 in enumerate(_o2[i:]):
                                if _oo2 == _oo1:
                                    obj = tokens[i: i + j + 1]
                                    _object = ''.join([oi.lstrip("##") for oi in obj])
                                    _object = ' '.join(_object.split('[unused1]'))
                                    _object = _object[:len(_object) - 1]
                                    _predicate = id2predicate[rel_id]
                                    R.append((_subject, _predicate, _object))
                                    break
    return set(R)

def partial_match(pred_set, gold_set):
    pred = {(i[0].split(' ')[0] if len(i[0].split(' ')) > 0 else i[0], i[1], i[2].split(' ')[0] if len(i[2].split(' ')) > 0 else i[2]) for i in pred_set}
    gold = {(i[0].split(' ')[0] if len(i[0].split(' ')) > 0 else i[0], i[1], i[2].split(' ')[0] if len(i[2].split(' ')) > 0 else i[2]) for i in gold_set}
    return pred, gold

def get_rel(pred_set,gold_set):
    pred = {(i[1]) for i in pred_set}
    gold = {(i[1]) for i in gold_set}
    return set(pred), set(gold)

def get_ent(pred_set,gold_set):
    pred = {(i[0], i[2]) for i in pred_set}
    gold = {(i[0], i[2]) for i in gold_set}
    return set(pred), set(gold)

def get_ent1_r(pred_set,gold_set):
    pred = {(i[0], i[1]) for i in pred_set}
    gold = {(i[0], i[1]) for i in gold_set}
    return set(pred), set(gold)

def get_r_ent2(pred_set,gold_set):
    pred = {(i[1], i[2]) for i in pred_set}
    gold = {(i[1], i[2]) for i in gold_set}
    return set(pred), set(gold)

def evaluate(tokenizer, data, id2predicate, model):
    official_A, official_B, official_C = 1e-10, 1e-10, 1e-10
    manual_A, manual_B, manual_C = 1e-10, 1e-10, 1e-10

    results = []
    for d in tqdm(iter(data)):
        R = extract_items_bert(d['spo_list'], tokenizer, d['tokens'], id2predicate, model)
        official_T = set([tuple(i) for i in d['spo_list']])
        results.append({'text':' '.join(d['tokens']), 'predict':list(R), 'truth':list(official_T)})

        official_A += len(R & official_T)
        official_B += len(R)
        official_C += len(official_T)
    return 2 * official_A / (official_B + official_C), official_A / official_B, official_A / official_C, results


def other_evaluate(tokenizer, data, id2predicate, model):
    official_A, official_B, official_C = 1e-10, 1e-10, 1e-10
    manual_A, manual_B, manual_C = 1e-10, 1e-10, 1e-10

    results = []
    for d in tqdm(iter(data)):
        R = other_extract_items_bert(tokenizer, d['sentText'], id2predicate, model)
        official_T = set(tuple([i["em1Text"], i["label"], i["em2Text"]]) for i in d["relationMentions"])
        results.append({'text':''.join(d['sentText']), 'predict':list(R), 'truth':list(official_T)})

        Pred_triples_eval, Gold_triples_eval = partial_match(R, official_T)

        official_A += len(Pred_triples_eval & Gold_triples_eval)
        official_B += len(Pred_triples_eval)
        official_C += len(Gold_triples_eval)

    return 2 * official_A / (official_B + official_C), official_A / official_B, official_A / official_C, results

def other_evaluate_compare(tokenizer, data, id2predicate, model):
    official_A, official_B, official_C = 1e-10, 1e-10, 1e-10
    manual_A, manual_B, manual_C = 1e-10, 1e-10, 1e-10
    rel_A, rel_B, rel_C = 1e-10, 1e-10, 1e-10
    ent_A, ent_B, ent_C = 1e-10, 1e-10, 1e-10
    ent1_r_A, ent1_r_B, ent1_r_C = 1e-10, 1e-10, 1e-10
    r_ent2_A, r_ent2_B, r_ent2_C = 1e-10, 1e-10, 1e-10
    results = []
    for d in tqdm(iter(data)):
        R = other_extract_items_bert(tokenizer, d['sentText'], id2predicate, model)
        official_T = set(tuple([i["em1Text"], i["label"], i["em2Text"]]) for i in d["relationMentions"])
        results.append({'text':''.join(d['sentText']), 'predict':list(R), 'truth':list(official_T)})

        Pred_triples_eval, Gold_triples_eval = partial_match(R, official_T)
        official_A += len(Pred_triples_eval & Gold_triples_eval)
        official_B += len(Pred_triples_eval)
        official_C += len(Gold_triples_eval)
        # relation
        p_rel_eval, g_rel_eval = get_rel(Pred_triples_eval, Gold_triples_eval)
        rel_A += len(p_rel_eval & g_rel_eval)
        rel_B += len(p_rel_eval)
        rel_C += len(g_rel_eval)
        # entity
        p_ent_eval, g_ent_eval = get_ent(Pred_triples_eval, Gold_triples_eval)
        ent_A += len(p_ent_eval & g_ent_eval)
        ent_B += len(p_ent_eval)
        ent_C += len(g_ent_eval)

        p_ent1_r_eval, g_ent1_r_eval = get_ent1_r(Pred_triples_eval, Gold_triples_eval)
        ent1_r_A += len(p_ent1_r_eval & g_ent1_r_eval)
        ent1_r_B += len(p_ent1_r_eval)
        ent1_r_C += len(g_ent1_r_eval)

        p_r_ent2_eval, g_r_ent2_eval = get_r_ent2(Pred_triples_eval, Gold_triples_eval)
        r_ent2_A += len(p_r_ent2_eval & g_r_ent2_eval)
        r_ent2_B += len(p_r_ent2_eval)
        r_ent2_C += len(g_r_ent2_eval)

    return 2 * official_A / (official_B + official_C), official_A / official_B, official_A / official_C, results,\
           2 * rel_A / (rel_B + rel_C), rel_A / rel_B, rel_A / rel_C, 2 * ent_A / (ent_B + ent_C), ent_A / ent_B, ent_A / ent_C, \
           2 * ent1_r_A / (ent1_r_B + ent1_r_C), ent1_r_A / ent1_r_B, ent1_r_A / ent1_r_C, 2 * r_ent2_A / (r_ent2_B + r_ent2_C), r_ent2_A / r_ent2_B, r_ent2_A / r_ent2_C


def evaluate_compare(tokenizer, data, id2predicate, model):
    official_A, official_B, official_C = 1e-10, 1e-10, 1e-10
    manual_A, manual_B, manual_C = 1e-10, 1e-10, 1e-10
    rel_A, rel_B, rel_C = 1e-10, 1e-10, 1e-10
    ent_A, ent_B, ent_C = 1e-10, 1e-10, 1e-10
    ent1_r_A, ent1_r_B, ent1_r_C = 1e-10, 1e-10, 1e-10
    r_ent2_A, r_ent2_B, r_ent2_C = 1e-10, 1e-10, 1e-10
    results = []
    for d in tqdm(iter(data)):

        R = extract_items_bert(d['spo_list'], tokenizer, d['tokens'], id2predicate, model)
        official_T = set([tuple(i) for i in d['spo_list']])
        results.append({'text':' '.join(d['tokens']), 'predict':list(R), 'truth':list(official_T)})

        Pred_triples_eval, Gold_triples_eval = R, official_T
        official_A += len(Pred_triples_eval & Gold_triples_eval)
        official_B += len(Pred_triples_eval)
        official_C += len(Gold_triples_eval)
        # relation
        p_rel_eval, g_rel_eval = get_rel(Pred_triples_eval, Gold_triples_eval)
        rel_A += len(p_rel_eval & g_rel_eval)
        rel_B += len(p_rel_eval)
        rel_C += len(g_rel_eval)
        # entity
        p_ent_eval, g_ent_eval = get_ent(Pred_triples_eval, Gold_triples_eval)
        ent_A += len(p_ent_eval & g_ent_eval)
        ent_B += len(p_ent_eval)
        ent_C += len(g_ent_eval)

        p_ent1_r_eval, g_ent1_r_eval = get_ent1_r(Pred_triples_eval, Gold_triples_eval)
        ent1_r_A += len(p_ent1_r_eval & g_ent1_r_eval)
        ent1_r_B += len(p_ent1_r_eval)
        ent1_r_C += len(g_ent1_r_eval)

        p_r_ent2_eval, g_r_ent2_eval = get_r_ent2(Pred_triples_eval, Gold_triples_eval)
        r_ent2_A += len(p_r_ent2_eval & g_r_ent2_eval)
        r_ent2_B += len(p_r_ent2_eval)
        r_ent2_C += len(g_r_ent2_eval)

    return 2 * official_A / (official_B + official_C), official_A / official_B, official_A / official_C, results,\
           2 * rel_A / (rel_B + rel_C), rel_A / rel_B, rel_A / rel_C, 2 * ent_A / (ent_B + ent_C), ent_A / ent_B, ent_A / ent_C, \
           2 * ent1_r_A / (ent1_r_B + ent1_r_C), ent1_r_A / ent1_r_B, ent1_r_A / ent1_r_C, 2 * r_ent2_A / (r_ent2_B + r_ent2_C), r_ent2_A / r_ent2_B, r_ent2_A / r_ent2_C




    


