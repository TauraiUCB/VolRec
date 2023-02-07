import pandas as pd
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime


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

def network():
    data = set_up()
    network = pd.crosstab(data['UserId'], data['ItemId'])
    network = network.apply(lambda row: row/row.sum(), axis = 1) # normalized/proobability P(o|v)
    row_id = network.index
    neigh = cosine_similarity(network)
    neigh[np.diag_indices_from(neigh)]= -1
    _neigh = pd.DataFrame(neigh, index = row_id, columns = row_id)
    net = pd.DataFrame(_neigh.apply(lambda x: list(_neigh.columns[np.array(x)
                .argsort()[::-1][:10]]),axis=1)
                .to_list(), columns = ['a','b','c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])

    net['UserId'] = row_id
    net = pd.melt(net, id_vars=['UserId'])
    net['Weight'] = 1
    net = net.rename(columns={'UserId': 'Followee', 'value': 'Follower'})
    net = net[['Follower', 'Followee', 'Weight']]
    net.to_csv('volunteer_network.tsv')

if __name__ == '__main__':
    network()
