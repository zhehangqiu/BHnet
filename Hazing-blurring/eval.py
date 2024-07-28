import os
import torch
from torchvision.transforms import functional as F
from utils import Adder
from data import test_dataloader
from data import valid_dataloader
from skimage.metrics import peak_signal_noise_ratio
import torch.nn.functional as f


def _eval(model, args):
    # 从保存的模型状态字典中加载模型参数
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict['model'])
    # 检查是否有可用的 GPU，选择合适的设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 创建用于测试的数据加载器
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
    # 释放 CUDA 缓存，清理 GPU 内存
    torch.cuda.empty_cache()
    # 将模型设置为评估（推理）模式
    model.eval()
    # 缩放因子，用于后续图像处理
    factor = 8

    # 禁用梯度计算，因为在评估阶段不需要计算梯度
    with torch.no_grad():
        # 用于累积 PSNR 的辅助函数
        psnr_adder = Adder()
        # 遍历测试数据集
        for iter_idx, data in enumerate(dataloader):
            # 从数据中获取输入图像、标签图像和名称
            input_img, label_img, name = data
            input_img = input_img.to(device)

            # 计算输入图像的高度和宽度
            h, w = input_img.shape[2], input_img.shape[3]
            # 将高度和宽度调整为 factor 的倍数，同时进行反射填充
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor * factor)
            padh = H-h if h%factor != 0 else 0
            padw = W-w if w%factor != 0 else 0
            input_img = f.pad(input_img, (0, padw, 0, padh), 'reflect')
            # 使用模型进行推理，获取预测结果
            pred = model(input_img)[2]
            pred = pred[:, :, :h, :w]
            # 将预测结果限制在 [0, 1] 的范围内
            pred_clip = torch.clamp(pred, 0, 1)
            # 转换为 NumPy 数组，以便后续计算 PSNR
            pred_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()
            # 如果设置了保存图像的标志，则保存预测结果
            if args.save_image:
                save_name = os.path.join(args.result_dir, name[0])
                pred_clip += 0.5 / 255
                pred = F.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred.save(save_name)
            # 计算并打印每个迭代的 PSNR
            psnr = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
            psnr_adder(psnr)
            print('%d iter PSNR: %.2f ' % (iter_idx + 1, psnr))

        # 打印平均 PSNR
        print('==========================================================')
        print('The average PSNR is %.2f dB' % (psnr_adder.average()))


