import json
from tqdm import tqdm

def get_num_data(in_file):
    with open(in_file, 'r') as fi,  open('test_num_1.json', 'w') as f1, open('test_num_2.json', 'w') as f2, open('test_num_3.json', 'w') as f3, open('test_num_4.json', 'w') as f4, open('test_num_5.json', 'w') as f5:
        for l in tqdm(fi):
            a = json.loads(l)
            if not a['relationMentions']:
                continue
            triples = a['relationMentions']
            triples_num = len(triples)
            if triples_num == 1:
                f1.write(json.dumps(a) + '\n')
            elif triples_num == 2:
                f2.write(json.dumps(a) + '\n')
            elif triples_num == 3:
                f3.write(json.dumps(a) + '\n')
            elif triples_num == 4:
                f4.write(json.dumps(a) + '\n')
            else:
                f5.write(json.dumps(a) + '\n')

if __name__ == '__main__':
    file_dir = '../data/'
    file_name = 'test.json'    # test file
    get_num_data(file_dir+file_name)
