#cli入口

import os
# 获取当前文件所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.getcwd()

import argparse
from .train import train_model
from .evaluate import run_evaluation
from .test import test_output

def main():
    parser = argparse.ArgumentParser(description="简繁体转换工具")
    subparsers = parser.add_subparsers(dest="command")

    parser.add_argument("--direction", required=True)
    parser.add_argument("--mapping_path",default=None,help="mapping_path")   
    parser.add_argument("--src_file",default=None,help="输入向量")  
    parser.add_argument("--tgt_file",default=None,help="输出向量")
    parser.add_argument("--dataset",default=None,help="待测试的数据集")   #data/testset.s.txt   data/testset.t.txt
    parser.add_argument("--dataset1",default=None,help="评估的数据集")   #data/testset.s.txt   data/testset.t.txt

    train_p = subparsers.add_parser("train", help="训练模型")
    train_p.add_argument("--epochs", type=int, default=1)
    train_p.add_argument("--batch-size", type=int, default=1)
    train_p.add_argument("--file",default="model") #模型保存路径
    train_p.add_argument("--data",default=None)#模型训练数据集

    eval_p = subparsers.add_parser("eval", help="评估模型")
    eval_p.add_argument("--model_path", help="模型文件")   #model/param-t-sg.ckpt     model/param-s-tg.ckpt

    test_p = subparsers.add_parser("test", help="模型输出")
    test_p.add_argument("--model_path", help="模型文件")   #model/param-t-sg.ckpt     model/param-s-tg.ckpt
    test_p.add_argument("--output",help="结果保存文件")   #data/testset.s.txt   data/testset.t.txt
     


    args = parser.parse_args()
    # 定义默认配置
    defaults = {
    "s2t": {
        "src_file": "data/S2T/vectors-cn.txt",
        "tgt_file": "data/S2T/vectors-tw.txt",
        "mapping_path": "data/S2T/best_mapping.pth",
        "data": "data/trainset.t.txt",
        "dataset":"data/testset.s.txt",
        "dataset1":"data/testset.t.txt",
        "model_path":"model/param-s-tg.ckpt",
        "output":"output"
    },                  
    "t2s": {
        "src_file": "data/T2S/vectors-tw.txt",
        "tgt_file": "data/T2S/vectors-cn.txt",
        "mapping_path": "data/T2S/best_mapping.pth",
        "data": "data/trainset.s.txt",
        "dataset":"data/testset.t.txt",
        "dataset1":"data/testset.s.txt",
        "model_path":"model/param-t-sg.ckpt",
        "output":"output"
    }                   
    }
    
    
    if args.direction =="s2t" or args.direction=="t2s":
        # 根据 direction 应用默认值（只有在 None 才赋值）
        
        cfg = defaults.get(args.direction, {})
        for key, value in cfg.items():
            if not hasattr(args, key) or getattr(args, key) in (None, ""):
                
                value=os.path.join(BASE_DIR,value)
                setattr(args, key, value)
    else:
        if args.src_file is None or args.tgt_file is None or args.mapping_path is None:
            raise ValueError("缺少重要参数文件！！！")
        if args.command=="test"  and args.dataset is None:
            raise ValueError("数据集缺失！！！")
        if args.command=="train" and args.data is None:
            raise ValueError("训练数据集缺失！！！")
    
    if args.command == "train":
        train_model(args.direction, args.epochs, args.batch_size,args.src_file,args.tgt_file,args.mapping_path,args.data,args.file)
    elif args.command == "eval":
        run_evaluation(args.direction, args.src_file,args.tgt_file,args.mapping_path, args.model_path,args.dataset,args.dataset1)
    elif args.command=="test":
        test_output(args.direction, args.src_file,args.tgt_file,args.mapping_path,args.model_path,args.dataset,args.output)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
