import pandas as pd
import numpy as np
import math
import argparse
import random
from collections import Counter



NETWORK_FILE = './data/volunteer_network.tsv'
max_length = 30




PATH_TO_DATA = './Pioneer/'

#load the org user data


def set_up():
    org_user = pd.read_csv('data/organizer_user_data.csv')
    org_user = org_user[org_user['organizer id'] != -1] #drop the -1 app test data
    org_user.rename(
        columns={'organizer id': 'org_id', 'Unnamed: 0': 'count', 'user id': 'user_id', 'issued time': 'Timestamp',
                 'districts': 'Location'}, inplace=True)
    org_user = org_user.drop(['count'], axis=1)
    _mapping = org_user.Location.unique()
    _mapping_dict = dict(zip(_mapping, range(len(_mapping))))
    _org_user = org_user.applymap(lambda x: _mapping_dict.get(x) if x in _mapping_dict else x)
    _org_user = _org_user.rename(columns={'Timestamp': 'dates'})
    _org_user['ts'] = _org_user['dates'].apply(lambda x: pd.Timestamp(x))
    _org_user['Timestamp'] = _org_user.ts.values.astype(np.int64) // 10 ** 9
    _org_user = _org_user.drop(['dates'], axis=1)
    _org_user = _org_user.drop(['ts'], axis=1)
    _org_user = _org_user.rename(columns={'user_id': 'UserId', 'org_id': 'ItemId'})
    _org_user = _org_user[['UserId', 'ItemId', 'Location', 'Timestamp']]
    return _org_user

def process_rating(day): # segment session in every $day days.
    df = set_up()
    df = df[df['Location'].between(1,6,inclusive=True)]
    min_timestamp = df['Timestamp'].min()
    time_id = [int(math.floor((t-min_timestamp) / (86400*day))) for t in df['Timestamp']]
    df['TimeId'] = time_id
    session_id = [str(uid)+'_'+str(tid) for uid, tid in zip(df['UserId'], df['TimeId'])]
    df['SessionId'] = session_id
    print('Statistics of user ratings:')
    print('\tNumber of total events: {}'.format(len(df)))
    print('\tNumber of users: {}'.format(df.UserId.nunique()))
    print('\tNumber of items: {}'.format(df.ItemId.nunique()))
    print('\tAverage ratings per user:{}'.format(df.groupby('UserId').size().mean()))
    return df

def process_social(): # read in social network.
    net = pd.read_csv(NETWORK_FILE, dtype={0:str, 1: str})
    net.drop_duplicates(subset=['Follower', 'Followee'], inplace=True)
    friend_size = net.groupby('Follower').size()
    #net = net[np.in1d(net.Follower, friend_size[friend_size>=5].index)]
    print('Statistics of volunteer network:')
    print('\tTotal user in volunteer network:{}.\n\tTotal edges(links) in volunteer network:{}.'.format(\
        net.Follower.nunique(), len(net)))
    print('\tAverage number of friends for volunteers: {}'.format(net.groupby('Follower').size().mean()))
    return net

def reset_id(data, id_map, column_name='UserId'):
    mapped_id = data[column_name].map(id_map)
    data[column_name] = mapped_id
    if column_name == 'UserId':
        session_id = [str(uid)+'_'+str(tid) for uid, tid in zip(data['UserId'], data['TimeId'])]
        data['SessionId'] = session_id
    return data

def split_data(day): #split data for training/validation/testing.
    df_data = process_rating(day)
    for i in range(0, len(df_data.columns)):
        df_data.iloc[:, i] = pd.to_numeric(df_data.iloc[:, i], errors='ignore')
        # errors='ignore' lets strings remain as 'non-null objects'
    df_net = process_social()
    for i in range(0, len(df_net.columns)):
        df_net.iloc[:, i] = pd.to_numeric(df_net.iloc[:, i], errors='ignore')
        # errors='ignore' lets strings remain as 'non-null objects'
    df_net = df_net.loc[df_net['Follower'].isin(df_data['UserId'])].drop_duplicates() #.unique())]
    df_net = df_net.loc[df_net['Followee'].isin(df_data['UserId'])].drop_duplicates() #.unique())]
    df_data = df_data.loc[df_data['UserId'].isin(df_net.Follower)].drop_duplicates() #.unique())]
    
    #restrict session length in [2, max_length]. We set a max_length because too long sequence may come from a fake user.
    df_data = df_data[df_data['SessionId'].groupby(df_data['SessionId']).transform('size')>1]
    df_data = df_data[df_data['SessionId'].groupby(df_data['SessionId']).transform('size')<=max_length]
    #length_supports = df_data.groupby('SessionId').size()
    #df_data = df_data[np.in1d(df_data.SessionId, length_supports[length_supports<=max_length].index)]
    
    # split train, test, valid.
    tmax = df_data.TimeId.max()
    session_max_times = df_data.groupby('SessionId').TimeId.max()
    session_train = session_max_times[session_max_times < tmax - 10].index
    session_holdout = session_max_times[session_max_times >= tmax - 10].index
    train_tr = df_data[df_data['SessionId'].isin(session_train)] 
    holdout_data = df_data[df_data['SessionId'].isin(session_holdout)] 
    
    print('Number of train/test: {}/{}'.format(len(train_tr), len(holdout_data)))
   
    train_tr = train_tr[train_tr['ItemId'].groupby(train_tr['ItemId']).transform('size')>20]
    train_tr = train_tr[train_tr['SessionId'].groupby(train_tr['SessionId']).transform('size')>1]
    
    print('Item size in train data: {}'.format(train_tr['ItemId'].nunique()))
    train_item_counter = Counter(train_tr.ItemId)
    to_predict = Counter(el for el in train_item_counter.elements() if train_item_counter[el] >= 50).keys()
    print('Size of to predict: {}'.format(len(to_predict)))
    
    # split holdout to valid and test.
    holdout_cn = holdout_data.SessionId.nunique()
    holdout_ids = holdout_data.SessionId.unique()
    np.random.shuffle(holdout_ids)
    valid_cn = int(holdout_cn * 0.5)
    session_valid = holdout_ids[0: valid_cn]
    session_test = holdout_ids[valid_cn: ]
    valid = holdout_data[holdout_data['SessionId'].isin(session_valid)]
    test = holdout_data[holdout_data['SessionId'].isin(session_test)]

    valid = valid[valid['ItemId'].isin(to_predict)]
    valid = valid[valid['SessionId'].groupby(valid['SessionId']).transform('size')>1]
    
    test = test[test['ItemId'].isin(to_predict)]
    test = test[test['SessionId'].groupby(test['SessionId']).transform('size')>1]

    total_df = pd.concat([train_tr, valid, test])
    df_net = df_net.loc[df_net['Follower'].isin(total_df['UserId'].unique())]
    df_net = df_net.loc[df_net['Followee'].isin(total_df['UserId'].unique())]
    user_map = dict(zip(total_df.UserId.unique(), range(total_df.UserId.nunique()))) 
    item_map = dict(zip(total_df.ItemId.unique(), range(1, 1+total_df.ItemId.nunique()))) 
    with open('user_id_map.tsv', 'w') as fout:
        for k, v in user_map.items():
            fout.write(str(k) + '\t' + str(v) + '\n')
    with open('item_id_map.tsv', 'w') as fout:
        for k, v in item_map.items():
            fout.write(str(k) + '\t' + str(v) + '\n')
    num_users = len(user_map)
    num_items = len(item_map)
    reset_id(total_df, user_map)
    reset_id(train_tr, user_map)
    reset_id(valid, user_map)
    reset_id(test, user_map)
    reset_id(df_net, user_map, 'Follower')
    reset_id(df_net, user_map, 'Followee')
    reset_id(total_df, item_map, 'ItemId')
    reset_id(train_tr, item_map, 'ItemId')
    reset_id(valid, item_map, 'ItemId')
    reset_id(test, item_map, 'ItemId')
    
    print ('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tAvg length: {}'.format(len(train_tr), train_tr.SessionId.nunique(), train_tr.ItemId.nunique(), train_tr.groupby('SessionId').size().mean()))
    print ('Valid set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tAvg length: {}'.format(len(valid), valid.SessionId.nunique(), valid.ItemId.nunique(), valid.groupby('SessionId').size().mean()))
    print ('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}\n\tAvg length: {}'.format(len(test), test.SessionId.nunique(), test.ItemId.nunique(), test.groupby('SessionId').size().mean()))
    user2sessions = total_df.groupby('UserId')['SessionId'].apply(set).to_dict()
    user_latest_session = []
    for idx in range(num_users):
        sessions = user2sessions[idx]
        latest = []
        for t in range(tmax+1):
            if t == 0:
                latest.append('NULL')
            else:
                sess_id_tmp = str(idx) + '_' + str(t-1)
                if sess_id_tmp in sessions:
                    latest.append(sess_id_tmp)
                else:
                    latest.append(latest[t-1])
        user_latest_session.append(latest)
    
    train_tr.to_csv('train.tsv', sep='\t', index=False)
    valid.to_csv('valid.tsv', sep='\t', index=False)
    test.to_csv('test.tsv', sep='\t', index=False)
    df_net.to_csv('adj.tsv', sep='\t', index=False)
    with open('latest_sessions.txt', 'w') as fout:
        for idx in range(num_users):
            fout.write(','.join(user_latest_session[idx]) + '\n')


if __name__ == '__main__':
    day = 7
    split_data(day)
