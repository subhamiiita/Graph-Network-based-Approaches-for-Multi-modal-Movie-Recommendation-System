import helpers
import random
import numpy as np
import sys


from sklearn.metrics import mean_squared_error as mse

import model_gcn_ww as neural_network





movie_enc= helpers.load_pkl("../objs/enc_movie.obj")
user_enc= helpers.load_pkl("../objs/enc_user_final.obj")





movie_shape= len(movie_enc.get(list(movie_enc.keys())[0]))
user_shape= len(user_enc.get(list(user_enc.keys())[0]))

batch_size= 64



def shuffle_single_epoch(ratings):
    data_copied= ratings.copy()
    random.shuffle(data_copied)
    return data_copied


def normalize(rate):
    return rate/5
def de_normalize(rate):
    return rate*5



def copy_to_fix_shape(movie, user, rate, req_len):
    assert len(movie)==len(user)==len(rate)
    new_movie, new_user, new_rate= movie[:], user[:], rate[:]
    while len(new_movie)<req_len:
        r_ind= random.randint(0, len(movie)-1)
        new_movie.append(movie[r_ind])
        new_user.append(user[r_ind])
        new_rate.append(rate[r_ind])
    return new_movie, new_user, new_rate


def get_nth_batch(ratings, n, batch_size= batch_size, take_entire_data= False):
    users= []
    movies= []
    rates= []
    if take_entire_data:
        slice_start= 0
        slice_end= len(ratings)
    else:
        if (n+1)*batch_size>len(ratings):
            print("OUT OF RANGE BATCH ID")
        slice_start= n*batch_size
        slice_end= (n+1)*batch_size
    for user_id, movie_id, rate in ratings[slice_start: slice_end]:
        if user_enc.get(user_id) is None or movie_enc.get(movie_id) is None:
            continue
        users.append(user_enc.get(user_id))
        movies.append(movie_enc.get(movie_id))
        rates.append(normalize(rate))

    if not take_entire_data:
        movies, users, rates= copy_to_fix_shape(movies, users, rates, batch_size)
    users= np.array(users)
    movies= np.array(movies)
    rates= np.array(rates)

    return movies, users, rates



def train(model, data, test_data= None, no_of_epoch= 32):
    total_batches_train= int(len(data)/batch_size)
    for epoch in range(no_of_epoch):
        print("\n\n---- EPOCH: ", epoch, "------\n\n")
        data= shuffle_single_epoch(data)
        for batch_id in range(total_batches_train):
            print("Epoch: ", epoch+1, " Batch: ", batch_id)
            movies, users, rates= get_nth_batch(data, batch_id)
            model.fit([movies, users], rates, batch_size=batch_size, epochs=1, verbose=2)
        if test_data is not None:
            test(model, test_data, take_entire_data=False, save=True)



lest_rmse= float("inf")
def test(model, data, save= True, take_entire_data= True):
    if take_entire_data:
        movies, users, res_true= get_nth_batch(data, 0, take_entire_data=take_entire_data)
        res_pred= model.predict([movies, users], batch_size=batch_size)
        res_true= np.array(res_true)
        res_pred= np.array(res_pred).reshape(-1)
        assert len(res_true)==len(res_pred)
    else:
        total_batches_test=int(len(data)/batch_size)
        res_true, res_pred= np.array([]), np.array([])
        for batch_id in range(total_batches_test+1):
            movies, users, rates= get_nth_batch(data, batch_id)
            pred= model.predict([movies, users], batch_size=batch_size)
            pred= pred.reshape(-1)
            assert len(rates)==len(pred)
            res_true= np.concatenate([res_true, rates])
            res_pred= np.concatenate([res_pred, pred])
    y_true= de_normalize(res_true)
    y_pred= de_normalize(res_pred)
    # res_pred= np.array([round(x) for x in res_pred])
    for x in range(200):
        print(y_true[x], " : ", y_pred[x])

    rmse= calc_rms(y_true, y_pred)
    y_pred= np.array([round(x) for x in y_pred])
    rmse_n= calc_rms(y_true, y_pred)
    print("rmse: ", rmse, " rmse_norm: ", rmse_n)
    global lest_rmse
    if save and lest_rmse>rmse:
        lest_rmse= rmse
        helpers.save_model(model)

def calc_rms(t, p):
    return mse(t, p, squared=False)


def train_test_ext(train_obj, test_obj):
    model= neural_network.make_model(movie_shape, user_shape, window_size=batch_size)
    train(model, data=train_obj, test_data=test_obj)
    test(model, data= test_obj, save= False, take_entire_data=False)


def test_saved(saved_model, test_file_path):
    import keras
    model= keras.models.load_model(saved_model)
    ratings_test_path= helpers.path_of(test_file_path)
    test_obj= helpers.load_pkl(ratings_test_path)
    test(model, data= test_obj, take_entire_data=True)


if __name__=="__main__":

    if len(sys.argv) in [2,3]:
        model_loc= sys.argv[1]
        if model_loc.split(".")[-1]!="h5":
            print("Tried to load model that is not h5 format")
            exit()
        if len(sys.argv)==3:
            test_file_loc= sys.argv[2]
            if test_file_loc.split(".")[-1]!="obj":
                print("Test file must be in .obj format")
        else:
            test_file_loc= "../liv_data/objs/splits/u1.test.obj"

        test_saved(model_loc, test_file_loc)
    
    else:
        train_obj= helpers.load_pkl("../liv_data/objs/splits/u1.train.obj")
        test_obj= helpers.load_pkl("../liv_data/objs/splits/u1.test.obj")

        train_test_ext(train_obj, test_obj)
