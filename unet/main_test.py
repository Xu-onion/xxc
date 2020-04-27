from model import *
from data import *
 
# python main_test.py
"""
注：
	A: target_size()为图片尺寸，要求测试集图像尺寸设置和model输入图像尺寸保持一致，
		如果不设置图片尺寸，会对输入图片做resize为处理，输入网络和输出图像尺寸默认均为（256,256），
	B: 且要求图片位深为8位，24/32的会报错！！
	C: 测试集数据名称需要设置为：0.png……
	D：model.predict_generator( ,n, ):n为测试集中样本数量，需要手动设置，不然会报错！！
"""
# 输入测试数据集，
testGene = testGenerator("data/membrane/test",target_size = (256,256)) # data
 
# 导入模型
model = unet(input_size = (256,256,1)) # model
 
# 导入训练好的模型
model.load_weights("unet_membrane.hdf5")
 
# 预测数据
results = model.predict_generator(testGene,20,verbose=1) # keras
print(results)
saveResult("data/membrane/kiki",results) # data
print("over")
