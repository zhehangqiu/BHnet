import os
import torch
import argparse
from torch.backends import cudnn
from models.BHNet import build_net
from train import _train
from eval import _eval

from valid import _valid
import shutil

def main(args):
    cudnn.benchmark = True

    if not os.path.exists('results/'):
        os.makedirs(args.model_save_dir)
    if not os.path.exists('results/' + args.model_name + '/'):
        os.makedirs('results/' + args.model_name + '/')
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    mode = [args.mode, args.data]
    model = build_net(mode)
    print(model)

    if torch.cuda.is_available():
        model.cuda()
    if args.mode == 'train':
        _train(model, args)

    if args.mode == 'test':
        _eval(model, args)
        #eval1(model, args)

if __name__ == '__main__':
#argparse 是 Python 的一个标准库模块，用于解析命令行参数。它提供了一种简单而灵活的方式，让开发者能够轻松地编写用户友好的命令行界面。
# 通过 argparse，你可以定义你的命令行工具接受哪些参数，以及这些参数的类型、默认值等信息。
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', default='BHNet', type=str)

    parser.add_argument('--data_dir', type=str, default='GOPRO')

    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)
    parser.add_argument('--data', type=str, default='GOPRO', choices=['GOPRO', 'HIDE', 'RSBlur'])

    # Train
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=1500)
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--num_worker', type=int, default=8)
    parser.add_argument('--save_freq', type=int, default=50)
    parser.add_argument('--valid_freq', type=int, default=50)
    parser.add_argument('--resume', type=str, default='')

    # Test
    parser.add_argument('--test_model', type=str, default='GOPRO.pkl')
    parser.add_argument('--save_image', type=bool, default=False, choices=[True, False])
    #参数添加关键
    args = parser.parse_args()
    args.model_save_dir = os.path.join('results/', args.model_name, args.data, 'Training-Results/')
    args.result_dir = os.path.join('results/', args.model_name, args.data)


    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
# 复制 layers.py 到目标文件夹
    #shutil.copy('models/layers.py', args.model_save_dir)

# 复制 BHNet.py 到目标文件夹
    #shutil.copy('models/BHNet.py', args.model_save_dir)

# 复制 train.py 到目标文件夹
    #shutil.copy('train.py', args.model_save_dir)

# 复制 main.py 到目标文件夹

    #shutil.copy('main.py', args.model_save_dir)
    #command = 'cp ' + 'models/layers.py ' + args.model_save_dir
    #os.system(command)
    #command = 'cp ' + 'models/BHNet.py ' + args.model_save_dir
    #os.system(command)
    #command = 'cp ' + 'train.py ' + args.model_save_dir
    #os.system(command)
    #command = 'cp ' + 'main.py ' + args.model_save_dir
    #os.system(command)

    print(args)
    main(args)
