 
# main_train.py
from model import *
from data import *
 
# #os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# data_gen_args = dict() : 为keras自带的图像增强方法
data_gen_args = dict(rotation_range=0.2, #整数。随机旋转的度数范围。
                    width_shift_range=0.05, #浮点数、一维数组或整数
                    height_shift_range=0.05, #浮点数。剪切强度（以弧度逆时针方向剪切角度）。
                    shear_range=0.05, 
                    zoom_range=0.05, #浮点数 或 [lower, upper]。随机缩放范围
                    horizontal_flip=True, 
                    fill_mode='nearest') # {"constant", "nearest", "reflect" or "wrap"} 之一。默认为 'nearest'。输入边界以外的点根据给定的模式填充：		
# 建立测试集，样本和标签分别放在同一个目录下的两个文件夹中，文件夹名字为：'image','label'
#得到一个生成器，以batch=2的速率无限生成增强后的数据		
myGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None) # data
 
# 调用模型，默认模型输入图像size=(256,256,1),样本位深为8位
model = unet() # model
# 保存训练的模型参数到指定的文件夹，格式为.hdf5; 检测的值是'loss'使其更小。
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True) # keras
# 开始训练，steps_per_epoch为迭代次数，epochs：
model.fit_generator(myGene,steps_per_epoch=600,epochs=1,callbacks=[model_checkpoint]) # keras
 
