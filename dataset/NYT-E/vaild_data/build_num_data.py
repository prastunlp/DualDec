import json

def get_num_data(test_data):
    with open('test_num_1.json', 'w') as f1, open('test_num_2.json', 'w') as f2, open('test_num_3.json', 'w') as f3, open('test_num_4.json', 'w') as f4, open('test_num_5.json', 'w') as f5:
        test_1 = []
        test_2 = []
        test_3 = []
        test_4 = []
        test_5 = [] # triple_num >= 5
        for a in test_data:
            if not a['spo_list']:
                continue
            triples = a['spo_list']
            triples_num = len(triples)
            if triples_num == 1:
                test_1.append(a)
            elif triples_num == 2:
                test_2.append(a)
            elif triples_num == 3:
                test_3.append(a)
            elif triples_num == 4:
                test_4.append(a)
            else:
                test_5.append(a)

        f1.write(json.dumps(test_1))
        f2.write(json.dumps(test_2))
        f3.write(json.dumps(test_3))
        f4.write(json.dumps(test_4))
        f5.write(json.dumps(test_5))


if __name__ == '__main__':
    file_dir = '../data/'
    file_name = 'test.json'  # test file
    test_data = json.load(open(file_dir + file_name, errors='ignore'))
    get_num_data(test_data)

