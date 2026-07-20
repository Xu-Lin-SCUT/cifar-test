import pandas as pd
import config
import settings
import os
import pickle
import numpy as np


def unpickle(file):
    with open(file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    return batch

def load_data():
    args=config.parse_args()
    train_all_data=[]
    train_all_label=[]
    test_data=[]
    test_label=[]
    if settings.Do_process:
        loaddata_path=args.preprocessed_data_path
    else:
        loaddata_path=args.cifar_path
        train_path_list=[]
        train_path_list.append(os.path.join(loaddata_path,settings.name_batch1))
        train_path_list.append(os.path.join(loaddata_path,settings.name_batch2))
        train_path_list.append(os.path.join(loaddata_path,settings.name_batch3))
        train_path_list.append(os.path.join(loaddata_path,settings.name_batch4))
        train_path_list.append(os.path.join(loaddata_path,settings.name_batch5))
        test_path=os.path.join(loaddata_path,settings.name_testbatch)

        for train_paths in train_path_list:
            batch=unpickle(train_paths)
            image_data=batch[b'data']#(方括号里填键（key）名,str形式)
            label_data=batch[b'labels']
            train_all_data.append(image_data)
            train_all_label.extend(label_data)

        train_data=np.vstack(train_all_data)
        train_label=np.array(train_all_label)

        test_batch=unpickle(test_path)
        test_data=test_batch[b'data']
        test_label=np.array(test_batch[b'labels'])
    
        train_df=pd.DataFrame({
                'images':list(train_data),
                'labels':train_label
            })
        test_df=pd.DataFrame({
                'images':list(test_data),
                'labels':test_label
            })

    return train_df,test_df

def test():
    test_dict = unpickle('./dataset/cifar-10/data_batch_1')
    print(test_dict.keys())  
    print(type(test_dict[b'data']))   # 输出：<class 'numpy.ndarray'>
    print(type(test_dict[b'labels'])) # 输出：<class 'list'>
    print(test_dict[b'data'].shape)   # 输出：(10000, 3072)
    print(len(test_dict[b'labels']))  # 输出：10000

if __name__=='__main__':
    train_df, test_df= load_data()
    print(train_df.info())
    print(test_df.head(2))
    print(type(train_df['images'].iloc[0])) 