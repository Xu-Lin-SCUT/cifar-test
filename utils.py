import torch
import random
import os
import numpy as np
import config



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def accuracy(outputs,labels):
    """
    outputs应有的维度是[batch,10]
    labels应有的维度是[batch，]
    比较output的dim=1维度的最大值的idx与labels对于的数字
    """
    _,preds=torch.max(outputs,dim=1)
    return (preds==labels).float().mean().item()

def save_checkpoint(state, checkpoint_name):
    """保存模型和优化器状态"""
    args=config.parse_args()
    filename=os.path.join(args.checkpoint_path,f'{checkpoint_name}.pth')
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(checkpoint_name, model, optimizer=None,scheduler=None):
    """加载模型和优化器状态（用于恢复训练）"""
    args=config.parse_args()
    filename=os.path.join(args.checkpoint_path,f'{checkpoint_name}.pth')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(filename, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Scheduler restored, last_epoch: {scheduler.last_epoch}")
    epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    print(f"Checkpoint loaded from {filename}, resuming from epoch {epoch},best_acc:{best_acc:.4f}")
    return epoch,best_acc

def optimizer(parameters,lr):
    return torch.optim.Adam(parameters,lr=lr,weight_decay=1e-4)