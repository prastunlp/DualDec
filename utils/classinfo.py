
"""
relation/entity class process
"""

import numpy as np

def getclassname(classname_path):
    fi = open(classname_path,'r')
    classlst = []
    for line in fi.readlines():
        text = line.split('\t')
        classname = text[0]
        classlst.append(classname)
    fi.close()
    return classlst

def get_class_embedding(embedding_path,embdding_num,dim):
    fi = open(embedding_path, 'r')
    classname2id = dict()
    embeddings = np.zeros((embdding_num, dim))
    index = 0
    for line in fi.readlines():
        text = line.split(' ')
        classname = text[0]
        class_emb = text[1:]
        classname2id[classname] = index
        embeddings[index] = np.asarray(class_emb)
        index = index+1
    fi.close()
    return embeddings,classname2id

def load_class_embedding( wordtoidx, W_emb, class_name):
    print("load class embedding")
    name_list = [ k.lower().split(' ') for k in class_name]
    id_list = [ [ wordtoidx[i] for i in l] for l in name_list]
    value_list = [ [ W_emb[i] for i in l]    for l in id_list]
    value_mean = [ np.mean(l,0)  for l in value_list]
    return np.asarray(value_mean)
