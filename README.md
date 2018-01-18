数据准备
======
http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1trn_gnt.zip
http://www.nlpr.ia.ac.cn/databases/download/feature_data/HWDB1.1tst_gnt.zip

下载后解压到代码根目录


数据转换
======
若要把gnt文件全部转成图片，运行convert_data.py

训练与测试
======
主要代码在trainin_with_Chinese.py

若要训练，则运行app('train')，注释掉app('test')

若要训练，则运行app('test')，注释掉app('train')

详细讲解
======
参考我这篇文章

https://zhuanlan.zhihu.com/p/33071173

