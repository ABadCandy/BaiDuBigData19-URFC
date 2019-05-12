# BaiDuBigData19-URFC
my two networks solution with 0.67 accuracy for 9 classification.

主要为了用Pytorch复现 https://github.com/czczup/UrbanRegionFunctionClassification 这位大神的tensorflow实现的双分支网络baseline，
同时visit数据的转换和链接中visit2array.py效果一致，即转为7×26×24(天×周×小时)的特征矩阵。

不同点：
1. 数据预处理：除了简单的平移旋转降噪外还事先去除了全黑和全白图片、去雾、直方图均衡化等，剩余训练集图片数为39730张。
2. 图片网络：采用原始3×100×100的尺寸输入，利用imagenet的预训练模型se_resnext101_32x4d进行微调。
3. visit网络：输入尺寸为7×26×24，不过与tf版本不同之处为前者将24作为通道数，该版本将7作为通道数，这样长宽基本一致，可以利用cifar10或cifar100
的预训练ResNet系列模型进行微调，这里采用的是无预训练的dpn26网络。
4. 特征融合：图片网络最后一层的特征向量维度为256，visit网络最后特征维度为64，concat后为320，最后接9个节点的全连接层进行分类。



使用说明：
1. 预处理数据下载链接：https://pan.baidu.com/s/1Pil1LCesVy4m6Fsb02Ggow  提取码：q5vp 

在当前目录下新建data文件夹，将下载好的数据解压至该目录下，最后可以看到data文件夹下有npy,train.test三个子文件夹，
其中npy里存有转换好的train_visit和test_visit，train和test两个子文件夹里分别存放了筛选和预处理后的39730张训练图片，以及原始的1w张测试图片。

2. 执行 pip install -r requirements.txt 安装必要的运行库。

3. 执行 python multimain.py 即可开始训练和测试，其中一些超参数如epoch,batch_size等可在config.py中修改。

4. 等第3步执行完后会在sumbit文件夹下生成csv格式的预测结果，为了与提交系统要求保持一致需要再运行 python submission.py，最终在submit文件夹下得到submit.txt即可提交。


提交结果如图：
![image](https://github.com/ABadCandy/BaiDuBigData19-URFC/blob/master/images/132844245726548246.jpg)
