import numpy as np

np.random.seed(63)

n_item = 11667
n_test_sample = 50
test_all = []

with open('./test.txt', 'r') as f:

    for row in f.readlines():
        row = eval(row)

        test_sample = []
        test_sample.append(row[0][-1])

        while len(test_sample) < n_test_sample+1:
            
            cdd = np.random.randint(1, n_item+1)
            if cdd in row[0]:
                continue
            test_sample.append(cdd)

        test_all.append(test_sample)

np.save(f'./test_candidate_1_{n_test_sample}.npy', np.array(test_all))
