import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import config
import model
import utils

def predict_single_image(image_path, model_path='best_model'):
    """
    对单张图片进行预测，返回预测类别名称和置信度
    """
    # 1. 解析配置（主要是 checkpoint 路径）
    args = config.parse_args()
    
    # 2. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 3. 加载模型
    net = model.test_model().to(device)
    
    # 构建最佳模型的完整路径
    checkpoint_path = os.path.join(args.checkpoint_path, f'{model_path}.pth')
    if not os.path.exists(checkpoint_path):
        print(f"错误：找不到模型文件 {checkpoint_path}")
        return
    
    # 加载权重（这里使用 weights_only=False 忽略警告，或保持默认）
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    net.load_state_dict(checkpoint['model_state_dict'])
    print(f"成功加载模型，来自第 {checkpoint['epoch']} 轮，验证准确率 {checkpoint['best_acc']:.4f}")
    
    # 4. 切换到评估模式
    net.eval()
    
    # 5. 定义图像预处理（必须与训练时一致）
    transform = transforms.Compose([
        transforms.Resize((32, 32)),           # 调整到 32x32
        transforms.ToTensor(),                 # 转换为 [0,1] 的 Tensor
        # 注意：训练时没有做归一化（只是转为 float32），所以这里不添加 normalize
        # 如果训练时做了归一化，这里也需要添加相应的归一化
    ])
    
    # 6. 加载并预处理图片
    try:
        img = Image.open(image_path).convert('RGB')   # 确保是 RGB 三通道
    except Exception as e:
        print(f"无法打开图片 {image_path}: {e}")
        return
    
    input_tensor = transform(img)               # shape: (3, 32, 32)
    input_batch = input_tensor.unsqueeze(0)    # 增加 batch 维度 -> (1, 3, 32, 32)
    input_batch = input_batch.to(device)
    
    # 7. 推理
    with torch.no_grad():
        outputs = net(input_batch)              # (1, 10)
        probabilities = torch.softmax(outputs, dim=1)  # 转为概率
        confidence, pred_idx = torch.max(probabilities, dim=1)
    
    # 8. 获取类别名称（CIFAR-10 类别）
    cifar10_classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    pred_class = cifar10_classes[pred_idx.item()]
    confidence_val = confidence.item()
    
    print(f"预测类别: {pred_class} (索引 {pred_idx.item()})，置信度: {confidence_val:.4f}")
    return pred_class, confidence_val

if __name__ == '__main__':
    # 假设 test.jpg 位于项目根目录
    image_file = './test.jpg'
    predict_single_image(image_file, model_path='best_model')  # 可选 'newest_model'