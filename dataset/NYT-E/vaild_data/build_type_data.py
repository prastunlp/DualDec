import json

def is_normal_triple(triples):
    entities = set()
    for i, e in enumerate(triples):
        if i % 3 != 2: #relation index==2
            entities.add(e)
    return len(entities) == 2 * int(len(triples) / 3)

def is_epo_triples(triples):
    if is_normal_triple(triples):
        return False
    entity_pair = [tuple(triples[3 * i: 3 * i + 2]) for i in range(int(len(triples) / 3))]
    return len(entity_pair) != len(set(entity_pair))

def is_spo_triples(triples):  # epo is also spo
    if is_normal_triple(triples):
        return False
    entity_pair = [tuple(triples[3 * i: 3 * i + 2]) for i in range(int(len(triples) / 3))]
    entity_pair = set(entity_pair)
    entities = []
    for pair in entity_pair:
        entities.extend(pair)
    entities = set(entities)
    return len(entities) != 2 * len(entity_pair)

def load_data(test_data, normal_file, epo_file, seo_file):
    with open(normal_file, 'w') as f1, open(epo_file, 'w') as f2, open(seo_file, 'w') as f3:
        normal_ = []
        epo_ = []
        seo_ = []
        for line in test_data:
            triples = []
            for rel in line['spo_list']:
                triples.append(rel[0])
                triples.append(rel[2])
                triples.append(rel[1])
            if is_normal_triple(triples):
                normal_.append(line)
            if is_epo_triples(triples):
                epo_.append(line)
            if is_spo_triples(triples):
                seo_.append(line)
        print("normal_",len(normal_))
        print("epo_", len(epo_))
        print("seo_", len(seo_))
        f1.write(json.dumps(normal_))
        f2.write(json.dumps(epo_))
        f3.write(json.dumps(seo_))

if __name__ == '__main__':
    file_dir = '../data/'
    file_name = 'train.json'    # test file
    test_data = json.load(open(file_dir + file_name, errors='ignore'))
    output_normal = 'train_normal.json'
    output_epo = 'train_epo.json'
    output_seo = 'train_seo.json'
    load_data(test_data, output_normal, output_epo, output_seo)
