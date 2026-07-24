import config
import model
import utils
import pandas as pd
import torch
import settings
import numpy as np
import dataloader
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

def test():
    #设置device，参数
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args=config.parse_args()
    
    #加载测试集
    _, test_df = dataloader.load_data()
    test_dataset=dataloader.Cifar10_Dataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    #加载模型和参数
    test_model=model.test_model().to(device)
    checkpoint_path=os.path.join(args.checkpoint_path, 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        print("best_model.pth not found, please train first.")
        return
    checkpoint = torch.load(checkpoint_path, map_location=device)
    test_model.load_state_dict(checkpoint['model_state_dict'])

    test_model.eval()

    test_acc=0.0

    with torch.no_grad():
        for images,labels in test_loader:
            images=images.to(device)
            labels=labels.to(device)
            outputs=test_model(images)
            acc=utils.accuracy(outputs=outputs,labels=labels)
            test_acc+=acc*args.batch_size  # 乘以 batch 大小，恢复总正确数

    avg_acc = test_acc / len(test_dataset)
    print(f"Test Accuracy: {avg_acc:.4f}")


if __name__=='__main__':
    test()