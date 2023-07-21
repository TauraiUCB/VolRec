# Environment Settings
- Python 2.7.18
- Tensorflow 1.4.1
- Pandas 1.5.1
- Numpy 1.23.4
 
 # Usage
 For experimentation, we provide raw data in the 'data' file which should be used as follows: 
 1. We first run the ```vol_network.py``` where the folder containing the input file should be specified. The output is a volunteer network.
 2. The volunteer network and original file are inputs to the ```preprocess.py``` which has the following outputs:
 
- train.tsv: includes volunteer historical behaviors, which is organized by pandas.Dataframe in five fields (SessionId UserId ItemId Timestamps TimeId).
- valid.tsv: the same format as train.tsv, used for tuning hyperparameters.
- test.tsv: the same format as test.tsv, used for testing model.
- adj.tsv: includes links to volunteer networks, which is also organized by pandas.Dataframe in two fields (FromId, ToId).
- latest_session.tsv: serves as 'reference' to target volunteer. This file records all volunteers available session at each time slot. For example, at time slot t, it stores volunteer v's t-1 th session.
- user_id_map.tsv: maps original string of volunteer id to int.
- item_id_map.tsv: maps original string of organizer id to int.

- For shorthand and readability, we use *User* and *Item* to denote *Volunteer* and *Organiser*, respectively.

# Running the code
- All output files from ```preprocess.py``` should be put in a ```data``` folder.
- To run the code, run ```sh volrec.sh```

# Data
- The Pioneers Volunteer Data has over 2.5 million events, 80 043 unique volunteers and 7 391 organizers.
- Full Pioneers data is open sourced at https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/YUOOBB

# Reference
- Please cite our paper if you use this code or data in your own work
 ```
 @article{muvunza2023session,
  title={Session-based recommendation with temporal dynamics for large volunteer networks},
  author={Muvunza, Taurai and Li, Yang},
  journal={Journal of Intelligent Information Systems},
  pages={1--22},
  year={2023},
  publisher={Springer}
}
