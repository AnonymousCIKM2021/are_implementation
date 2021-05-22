import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix
from torch import LongTensor
from sklearn.model_selection import train_test_split
import gc


def load_data(
        dataset_name: str = 'douban',  # ml-100k, ml-1m or douban
        folder: str or Path = None,
        device: str = 'cuda',
        save_auxiliary: bool = False,
        seed: int = 1234,  # only necessary when dataset_name == 'ml-1m'
):
    if dataset_name == 'ml-100k':

        columns = 'UserID::MovieID::Rating::Timestamp'.lower().split('::')
        df_train = pd.read_csv(str(folder) + '/u1.base', sep='\t', header=None)
        df_val = pd.read_csv(str(folder) + '/u1.test', sep='\t', header=None)
        df_train.columns = columns
        df_val.columns = columns

        # Concatenate the dfs and enumerate both users and movies from 0 to len(set(users/movies))
        ratings_new = df_train.append(df_val)
        ratings_new['userid'], users_keys = ratings_new.userid.factorize()
        ratings_new['movieid'], movies_keys = ratings_new.movieid.factorize()
        # Split back
        ratings_new = ratings_new.values
        train = ratings_new[:len(df_train)]
        val = ratings_new[len(train):]
        # Create train and test as matrices of shape len(set(users)) x len(set(movies))
        shape = [int(ratings_new[:, 0].max() + 1), int(ratings_new[:, 1].max() + 1)]
        train = csr_matrix((train[:, 2].astype(int), (train[:, 0].astype(int), train[:, 1].astype(int))),
                           shape=shape).toarray()
        val = csr_matrix((val[:, 2].astype(int), (val[:, 0].astype(int), val[:, 1].astype(int))),
                         shape=shape).toarray()

        if save_auxiliary:
            np.save(folder + '/users_keys', users_keys)
            np.save(folder + '/movies_keys', movies_keys)
            np.save(folder + '/ratings_new', ratings_new)
            np.save(folder + '/train', train)
            np.save(folder + '/val', val)
            np.save(folder + '/ratings', ratings_new)

        movies_init = pd.read_csv(folder + '/u.item', sep='|', header=None, encoding="ISO-8859-1")
        movies = pd.DataFrame()
        movies['key'] = movies_init.iloc[:, 0]
        for i in range(5, 24):
            movies[f'genre_{i - 4}'] = movies_init.iloc[:, i] * (i - 4)

        df_item = pd.DataFrame(train.T)
        df_item['key'] = movies_keys
        df_item = df_item.merge(movies, on='key', how='left').drop(columns='key').values
        item_add_part = df_item[:, -movies.shape[1] + 1:]

        users_init = pd.read_csv(folder + '/u.user', sep='|', header=None, encoding="ISO-8859-1")
        users_init.columns = ['key', 'age', 'sex', 'occupation', 'index']
        users = pd.DataFrame()
        users['key'] = users_init.key
        users['is_male'] = (users_init.sex == 'M').astype(int)
        users['occup'], _ = users_init.occupation.factorize()

        df_user = pd.DataFrame(train)
        df_user['key'] = users_keys
        df_user = df_user.merge(users, on='key', how='left').drop(columns='key').values
        user_add_part = df_user[:, -users.shape[1] + 1:]

        train_tensor = LongTensor(train).to(device)
        val_tensor = LongTensor(val).to(device)
        user_features_tensor = LongTensor(user_add_part).to(device)
        item_features_tensor = LongTensor(item_add_part).to(device)

    elif dataset_name == 'ml-1m':

        ratings = pd.read_csv(folder + '/ratings.dat', sep='::', header=None)
        ratings.columns = 'UserID::MovieID::Rating::Timestamp'.lower().split('::')
        ratings['userid'], users_keys = ratings.userid.factorize()
        ratings['movieid'], movies_keys = ratings.movieid.factorize()

        # Split to train and val
        train, val = train_test_split(ratings.values, test_size=0.1, random_state=seed)
        shape = [int(ratings.iloc[:, 0].max() + 1), int(ratings.iloc[:, 1].max() + 1)]
        # Create train and test as matrices of shape len(set(users)) x len(set(movies))
        train = csr_matrix((train[:, 2].astype(int), (train[:, 0].astype(int), train[:, 1].astype(int))),
                           shape=shape).toarray()
        val = csr_matrix((val[:, 2].astype(int), (val[:, 0].astype(int), val[:, 1].astype(int))),
                         shape=shape).toarray()

        if save_auxiliary:
            np.save('users_keys', users_keys)
            np.save('movies_keys', movies_keys)
            np.save('train_new', train)
            np.save('val_new', val)
            np.save('ratings', ratings.values)

        movies_init = pd.read_csv(folder + '/movies.dat', sep='::', header=None, encoding="ISO-8859-1")
        movies = pd.DataFrame()
        movies['key'] = movies_init.iloc[:, 0]

        genres = np.unique(np.concatenate(movies_init.iloc[:, 2].apply(lambda x: x.split('|')).values))
        for i in range(1, 19):
            movies[f'genre_{i}'] = movies_init.iloc[:, 2].apply(lambda x: genres[i - 1] in x) * i

        df_item = pd.DataFrame(train.T)
        df_item['key'] = movies_keys
        df_item = df_item.merge(movies, on='key', how='left').drop(columns='key').values
        item_add_part = df_item[:, -movies.shape[1] + 1:]

        users_init = pd.read_csv(folder + '/users.dat', sep='::', header=None, encoding="ISO-8859-1")
        users_init.columns = ['key', 'sex', 'age', 'occupation', 'index']

        users = pd.DataFrame()
        users['key'] = users_init.key
        users['age'] = users_init.age.map({1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6})
        users['is_male'] = (users_init.sex == 'M').astype(int)
        users['occup'], _ = users_init.occupation.factorize()

        df_user = pd.DataFrame(train)
        df_user['key'] = users_keys
        df_user = df_user.merge(users, on='key', how='left').drop(columns='key').values
        user_add_part = df_user[:, -users.shape[1] + 1:]

        train_tensor = LongTensor(train).to(device)
        val_tensor = LongTensor(val).to(device)
        user_features_tensor = LongTensor(user_add_part).to(device)
        item_features_tensor = LongTensor(item_add_part).to(device)

    else:

        ratings = np.load(folder + '/douban.npy', allow_pickle=True)
        train = np.load(folder + '/otraining.npy', allow_pickle=True) * ratings
        val = np.load(folder + '/otest.npy', allow_pickle=True) * ratings

        train_tensor = LongTensor(train).to(device)
        val_tensor = LongTensor(val).to(device)
        user_features_tensor = None
        item_features_tensor = None

    gc.collect()
    return train_tensor, val_tensor, user_features_tensor, item_features_tensor


def load_constants(dataset_name: str = 'douban', model: str = 'are+feat'):

    constants_dict = {
        "douban": {
            "user_init_zero_emb": -0.15,
            "item_init_zero_emb": -0.1,
            "user_init_beta": 2.5,
            "item_init_beta": 0.5,
            "user_init_gamma": 0.6,
            "item_init_gamma": 0.47,
            "user_num_epochs": 10,
            "item_num_epochs": 4,
            "item_optimizer": ('AdamW', {'lr': 0.1}),
            "user_scheduler": False,
            "item_scheduler": False,
            "alpha": 0.95,
            "device": 'cuda'
        },
        "ml-100k": {
            'are+feat': {
                "user_num_epochs": 10,
                "item_num_epochs": 8,
                "alpha": 0.6,
                "device": 'cuda'
            },
            'are': {
                "user_num_epochs": 4,
                "item_num_epochs": 4,
                "alpha": 0.55,
                "item_init_zero_emb": 0.05,
                "device": 'cuda'
            }
        },
        "ml-1m": {
            'are+feat': {
                "user_num_epochs": 10,
                "item_num_epochs": 10,
                "alpha": 0.75,
                "user_scheduler": False,
                "device": 'cuda'
            },
            'are': {
                "user_num_epochs": 10,
                "item_num_epochs": 10,
                "alpha": 0.75,
                "item_init_means_add": [0.2, 0.1, 0.1],
                "user_scheduler": False,
                "device": 'cuda'
            }
        }
    }

    constants = constants_dict[dataset_name]
    if dataset_name != 'douban':
        constants = constants[model]

    return constants