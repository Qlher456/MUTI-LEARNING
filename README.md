# MUTI-LEARNING

# vgg16.py

vgg16基础网络在AIGC数据集上迭代100次

# 超参数

input_size = (224, 224)
batch_size = 128
learning_rate = 0.00001
epochs = 100
validation_split = 0.2
num_workers=4

Epoch 1/100
13/13 [==============================] - 26s 1s/step - loss: 0.7710 - accuracy: 0.5028 - val_loss: 0.6983 - val_accuracy: 0.5368
...
Epoch 100/100
13/13 [==============================] - 13s 779ms/step - loss: 0.3690 - accuracy: 0.8451 - val_loss: 0.8280 - val_accuracy: 0.5417

![image](https://github.com/user-attachments/assets/dc08d720-518f-414c-8812-319e923bc283)

