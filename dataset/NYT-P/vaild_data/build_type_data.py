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

def load_data(in_file, normal_file, epo_file, seo_file):
    with open(in_file, 'r') as f1,  open(normal_file, 'w') as f2, open(epo_file, 'w') as f3, open(seo_file, 'w') as f4:
        lines = f1.readlines()
        for line in lines:
            line = json.loads(line)
            triples = []
            for rel in line['relationMentions']:
                triples.append(rel['em1Text'])
                triples.append(rel['em2Text'])
                triples.append(rel['label'])
            if is_normal_triple(triples):
                f2.write(json.dumps(line) + '\n')
            if is_epo_triples(triples):
                f3.write(json.dumps(line) + '\n')
            if is_spo_triples(triples):
                f4.write(json.dumps(line) + '\n')

if __name__ == '__main__':
    file_dir = '../data/'
    file_name = 'test.json'    # test file
    output_normal = 'test_normal.json'
    output_epo = 'test_epo.json'
    output_seo = 'test_seo.json'
    load_data(file_dir+file_name, output_normal, output_epo, output_seo)
