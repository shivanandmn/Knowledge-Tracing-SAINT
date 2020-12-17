import config 

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import gc
from sklearn.model_selection import train_test_split 


class DKTDataset(Dataset):
  def __init__(self,group,n_skills,max_seq = 100):
    self.samples = group
    self.n_skills = n_skills
    self.max_seq = max_seq
    self.data = []

    for que,ans,res_time,exe_cat in self.samples:
        if len(que)>=self.max_seq:
            self.data.extend([(que[l:l+self.max_seq],ans[l:l+self.max_seq],res_time[l:l+self.max_seq],exe_cat[l:l+self.max_seq])\
            for l in range(len(que)) if l%self.max_seq==0])
        elif len(que)<self.max_seq and len(que)>10:
            self.data.append((que,ans,res_time,exe_cat))
        else :
            continue
  
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self,idx):
    content_ids,answered_correctly,response_time,exe_category = self.data[idx]
    seq_len = len(content_ids)

    q_ids = np.zeros(self.max_seq,dtype=int)
    ans = np.zeros(self.max_seq,dtype=int)
    r_time = np.zeros(self.max_seq,dtype=int)
    exe_cat = np.zeros(self.max_seq,dtype=int)

    if seq_len>=self.max_seq:
      q_ids[:] = content_ids[-self.max_seq:]
      ans[:] = answered_correctly[-self.max_seq:]
      r_time[:] = response_time[-self.max_seq:]
      exe_cat[:] = exe_category[-self.max_seq:]
    else:
      q_ids[-seq_len:] = content_ids
      ans[-seq_len:] = answered_correctly
      r_time[-seq_len:] = response_time
      exe_cat[-seq_len:] = exe_category
    
    target_qids = q_ids[1:]
    label = ans[1:] 

    input_ids = np.zeros(self.max_seq-1,dtype=int)
    input_ids = q_ids[:-1].copy()

    input_rtime = np.zeros(self.max_seq-1,dtype=int)
    input_rtime = r_time[:-1].copy()

    input_cat = np.zeros(self.max_seq-1,dtype=int)
    input_cat = exe_cat[:-1].copy()

    input = {"input_ids":input_ids,"input_rtime":input_rtime.astype(np.int),"input_cat":input_cat}

    return input,target_qids,label 



def get_dataloaders():              
    dtypes = {'timestamp': 'int64', 'user_id': 'int32' ,'content_id': 'int16',
                'answered_correctly':'int8',"prior_question_elapsed_time":"float32","task_container_id":"int16"}
    print("loading csv.....")
    train_df = pd.read_csv(config.TRAIN_FILE,dtype=dtypes,nrows=10000)
    print("shape of dataframe :",train_df.shape) 

    train_df = train_df[train_df.content_type_id==0]
    train_df.prior_question_elapsed_time /=1000
    train_df.prior_question_elapsed_time.fillna(300,inplace=True)
    train_df.prior_question_elapsed_time.clip(lower=0,upper=300,inplace=True)
    train_df.prior_question_elapsed_time = train_df.prior_question_elapsed_time.astype(np.int)
    
    
    train_df = train_df.sort_values(["timestamp"],ascending=True).reset_index(drop=True)
    skills = train_df.content_id.unique()
    n_skills = len(skills)
    n_cats = len(train_df.task_container_id.unique())+100
    print("no. of skills :",n_skills)
    print("no. of categories: ", n_cats)
    print("shape after exlusion:",train_df.shape)

    #grouping based on user_id to get the data supplu
    print("Grouping users...")
    group = train_df[["user_id","content_id","answered_correctly","prior_question_elapsed_time","task_container_id"]]\
                    .groupby("user_id")\
                    .apply(lambda r: (r.content_id.values,r.answered_correctly.values,r.prior_question_elapsed_time.values,r.task_container_id.values))
    del train_df
    gc.collect()

    print("splitting")
    train,val = train_test_split(group,test_size=0.2) 
    print("train size: ",train.shape,"validation size: ",val.shape)
    train_dataset = DKTDataset(train.values,n_skills=n_skills,max_seq = config.MAX_SEQ)
    val_dataset = DKTDataset(val.values,n_skills=n_skills,max_seq = config.MAX_SEQ)
    train_loader = DataLoader(train_dataset,
                          batch_size=config.BATCH_SIZE,
                          num_workers=2,
                          shuffle=True)
    val_loader = DataLoader(val_dataset,
                          batch_size=config.BATCH_SIZE,
                          num_workers=2,
                          shuffle=False)
    del train_dataset,val_dataset
    gc.collect()
    return train_loader, val_loader