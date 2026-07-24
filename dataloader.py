import pandas as pd
import config
import settings
import os
import pickle
import numpy as np
from torch.utils.data import Dataset,DataLoader
import torch
import torchvision.transforms as transforms

train_transform=transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop(32,padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                         std=[0.2023, 0.1994, 0.2010])
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], 
                         std=[0.2023, 0.1994, 0.2010])
])

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

class Cifar10_Dataset(Dataset):
    def __init__(self,df,transform=None):
        super().__init__()
        self.df=df
        self.transform=transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img = self.df.iloc[idx]['images'].reshape(3, 32, 32).astype(np.uint8)
        img = img.transpose(1, 2, 0)
        label=int(self.df.iloc[idx]['labels'])
        img = self.transform(img)
        
        return img, torch.tensor(label)

def test():
    test_dict = unpickle('./dataset/cifar-10/data_batch_1')
    print(test_dict.keys())  
    print(type(test_dict[b'data']))   # 输出：<class 'numpy.ndarray'>
    print(type(test_dict[b'labels'])) # 输出：<class 'list'>
    print(test_dict[b'data'].shape)   # 输出：(10000, 3072)
    print(len(test_dict[b'labels']))  # 输出：10000

if __name__=='__main__':

    # 1. 加载数据
    train_df, test_df = load_data()
    print(f"训练集样本数: {len(train_df)}, 测试集样本数: {len(test_df)}")
    
    # 2. 封装成 Dataset
    train_dataset = Cifar10_Dataset(train_df)
    test_dataset = Cifar10_Dataset(test_df)

    args=config.parse_args()
    # 3. 封装成 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # 4. 取一个批次验证形状
    for images, labels in train_loader:
        print(f"批次图片形状: {images.shape}")   # 预期: torch.Size([32, 3, 32, 32])
        print(f"批次标签形状: {labels.shape}")   # 预期: torch.Size([32])
        break