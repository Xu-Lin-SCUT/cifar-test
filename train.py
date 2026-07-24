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

def train():
#get hyperparameters
    args=config.parse_args()
    max_epoch=args.max_epoch
    lr=args.learning_rate
    batch_size=args.batch_size
#get device
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#control random
    utils.set_seed(42)
#get data
    train_df,test_df=dataloader.load_data()
    train_dataset=dataloader.Cifar10_Dataset(train_df,transform=dataloader.train_transform)
    test_dataset=dataloader.Cifar10_Dataset(test_df,transform=dataloader.test_transform)

#dataloader
    train_dataloader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=0,pin_memory=True)
    test_dataloader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,num_workers=0,pin_memory=True)
#load model,loss,optimizer
    CNN=model.test_model().to(device)
    criterion=torch.nn.CrossEntropyLoss()
    optimizer=utils.optimizer(CNN.parameters(),lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5,T_mult=2)

#initialize best acc(test)
    start_epoch = 1  # 默认从第 1 轮开始
    best_acc = 0  # 默认最佳准确率为 0
    if os.path.exists(os.path.join(args.checkpoint_path, 'newest_model.pth')):
        print("检测到已保存的最新模型")
    # 注意：load_checkpoint 需要能返回 best_acc，我们稍后修改 utils.py
        loaded_epoch,best_acc = utils.load_checkpoint('newest_model', CNN, optimizer,scheduler)
        start_epoch = loaded_epoch + 1  # 从下一轮开始
        print(f"从第 {start_epoch} 轮继续训练,最佳准确率为{best_acc}")
    else:
        print("未检测到保存的模型，从头开始训练")


#main loop
    for epoch in range(start_epoch,max_epoch+1):
        CNN.train()
        train_loss=0
        train_acc=0
#use tqdm
        pbar=tqdm(train_dataloader,desc=f"Epoch:{epoch}/{max_epoch}")
#batch loop
        for batch_idx,(images,labels) in enumerate(pbar):

            #print(f"图像均值: {images.mean():.3f}, 标准差: {images.std():.3f}")
#transform to GPU
            images,labels=images.to(device),labels.to(device)
#forward
            outputs=CNN(images)
#backward
            loss=criterion(outputs,labels)
#optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


#calculate data in this batch
            train_loss+=loss.item()
            batch_acc=utils.accuracy(outputs,labels)
            train_acc+=batch_acc


            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{batch_acc:.4f}'
            })
#batch loop ends
#calculate data in this epoch
        avg_train_loss = train_loss / len(train_dataloader)
        avg_train_acc = train_acc / len(train_dataloader)
        CNN.eval()  # 切换到评估模式
        test_loss = 0.0
        test_acc = 0.0
        with torch.no_grad():  # 关闭梯度计算（节省内存和计算）
            for images, labels in test_dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = CNN(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                test_acc += utils.accuracy(outputs, labels)
        
        # 计算测试集平均损失和准确率
        avg_test_loss = test_loss / len(test_dataloader)
        avg_test_acc = test_acc / len(test_dataloader)
        
        # ---------- 打印本轮结果 ----------
        print(f"Epoch [{epoch}/{max_epoch}] "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f} | "
              f"Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.4f}")
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': CNN.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_acc': best_acc,
            'scheduler_state_dict': scheduler.state_dict(),
        }
        print(f"Gamma 值: {CNN.attn.gamma.item():.4f}") 
        # ---------- 保存最佳模型 ----------
        if avg_test_acc > best_acc:
            best_acc = avg_test_acc
            utils.save_checkpoint(checkpoint, 'best_model')

        #----------- 保存最新模型  -----------
        utils.save_checkpoint(checkpoint,f'newest_model')
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"当前学习率: {current_lr:.6f}")

    print(f"训练完成！最佳测试准确率: {best_acc:.4f}")

if __name__=='__main__':
    train()