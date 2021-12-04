##  Effective Cascade Dual-Decoder Model for Joint Entity and RelationExtraction
  Effective Cascade Dual-Decoder Model for Joint Entity and RelationExtraction. [submitted]

## Requirements
All experiments are conducted with an NVDIA GeForce RTX 2080 Ti. 

The main requirements are:
- python = 3.6
- torch  = 1.1.0
- transformers = 3.5.1 (Online)

## Usage
### Training
1. Partial Match:

    python train.py --data_dir=dataset/WebNLG-P/data  --id=WebNLG-P --classemb_num=214 --entityclass_num=2 --relationclass_num=171

2. Exact Match:

    python othertrain.py --data_dir=dataset/WebNLG-E/data  --id=WebNLG-E --classemb_num=255 --entityclass_num=2 --relationclass_num=211
### Testing
1. Partial Match:

    python othereval.py --model_dir=./saved_models/WebNLG-P --data_dir=dataset/WebNLG-P/data

2. Exact Match:

    python eval.py --model_dir=./saved_models/WebNLG-E --data_dir=dataset/WebNLG-E/data

## Related Repo

Codes are adapted from the repositories of  [Joint Extraction of Entities and Relations Based on a Novel Decomposition](https://github.com/yubowen-ph/JointER) and [A Novel Cascade Binary Tagging Framework for Relational Triple Extraction](https://github.com/weizhepei/CasRel).

## Remark
This work experiment was completed in December 2020. We first submitted in ACL, which was accepted by Findings, and transferred to TKDE in July 2021, which was rejected in December 2021.
