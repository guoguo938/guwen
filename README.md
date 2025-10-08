An Unsupervised Framework for Adaptive Context-aware
Simplified-Traditional Chinese Character Conversion


tt目录下
    pip install .
T2S目录下：
    默认繁简转换（数据集和模型可指定）：
    T2S --direction t2s  --dataset 数据集 test --output output  
T2S --direction s2t train         (参数为默认值,可参照下方修改)



训练模型：
    数据集+向量 
    训练得到W转换矩阵:
        python muse_code/unsupervised --src_lang cn --tgt_lang tw --export txt --exp_path output --src_emb vectors-cn.txt  --tgt_emb vectors-tw.txt
    训练得到最终模型:
        T2S --direction s2t/t2s --src_file --tgt_file --mapping_path   train --file model    --data 训练集                      
    
eval：T2S --direction s2t/t2s   --src_file --tgt_file --mapping_path --dataset --dataset1 eval  --model_path model/param-s-tg.ckpt
test：T2S --direction s2t/t2s  --src_file --tgt_file --mapping_path --dataset  test --model_path  --output output
默认繁简转换：T2S --direction t2s  --dataset 数据集 test --output output --model_path

参数定义:
    src_file 输入向量
    tgt_file 输出向量
    mapping_path 映射矩阵 .pth
    dataset  待评估的数据集  dataset1 参考答案

    训练参数：
        data 模型训练数据集
        epoch,batch_size=1
        file 训练模型保存路径
    测试/转换输出：
        model_path 模型文件
        output  转换文件保存路径




