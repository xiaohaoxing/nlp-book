import pickle

pre_pkl_file = 'pre_cbow_params.pkl'
with open(pre_pkl_file, 'rb') as f:
    params = pickle.load(f)
    car_vec = params['word_vecs'][params['word_to_id']['car']]

    print(car_vec)
    print(len(car_vec))
    # for line in f.readlines():
    #     print(line)
