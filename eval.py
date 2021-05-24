"""
Run evaluation for Exact Match.
"""

import random
import argparse
import torch
import json
from models.remodel import REModel
from utils import torch_utils, helper, score
from transformers import BertTokenizer, BertConfig

model_name = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(model_name)

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default='./saved_models/WebNLG-E-01', help='Directory of the model.') #{ opt: [NYT-E-01 , WebNLG-E-01] }
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset/WebNLG-E/data') # {WebNLG-E  NYT-E}
# parser.add_argument('--data_dir', type=str, default='dataset/WebNLG/vaild_data') #{ WebNLG-E  NYT-E}
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")
# parser.add_argument('--dataset', type=str, default='test_num_5', help="Evaluate on dev or test.") # {test_epo | test_seo | test_normal | test_num_1 test_num_2 test_num_3 test_num_4 test_num_5}
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
model = REModel(opt)
model.load(model_file)

# load data
data_file = args.data_dir + '/{}.json'.format(args.dataset)
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
data = json.load(open(data_file,errors='ignore'))
id2predicate, predicate2id, id2subj_type, subj_type2id, id2obj_type, obj_type2id = json.load(open(opt['data_dir'] + '/schemas.json'))
id2predicate = {int(i):j for i,j in id2predicate.items()}

helper.print_config(opt)

#eval
f1, p, r, results = score.evaluate(tokenizer, data, id2predicate, model)
results_save_dir = opt['model_save_dir'] + '/best_{}_results.json'.format(args.dataset)
print("Dumping the best test results to {}".format(results_save_dir))
with open(results_save_dir, 'w') as fw:
    json.dump(results, fw, indent=4, ensure_ascii=False)

print("data_file: {}: p = {:.6f}, r = {:.6f}, f1 = {:.4f}".format(args.dataset, p, r, f1))
print("Evaluation ended.")

###eval triple element
# f1, p, r, results, rf1, rp, rr, ef1, ep, er, erf1, erp, err, ref1, rep, rer = score.evaluate_compare(tokenizer, data, id2predicate, model)
# results_save_dir = opt['model_save_dir'] + '/best_{}_results.json'.format(args.dataset)
# print("Dumping the best test results to {}".format(results_save_dir))
# with open(results_save_dir, 'w') as fw:
#     json.dump(results, fw, indent=4, ensure_ascii=False)
#
# print("data_file: {}: p = {:.6f}, r = {:.6f}, f1 = {:.4f}".format(args.dataset, p, r, f1))
# print("reldata_file: {}: p = {:.6f}, r = {:.6f}, f1 = {:.4f}".format(args.dataset, rp, rr, rf1))
# print("entdata_file: {}: p = {:.6f}, r = {:.6f}, f1 = {:.4f}".format(args.dataset, ep, er, ef1))
# print("ent1-r data_file: {}: p = {:.6f}, r = {:.6f}, f1 = {:.4f}".format(args.dataset, erp, err, erf1))
# print("r-ent2 data_file: {}: p = {:.6f}, r = {:.6f}, f1 = {:.4f}".format(args.dataset, rep, rer, ref1))
# print("Evaluation ended.")