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

# VIT.py

VIT模型在AIGC上迭代100次

# config.json

{

  "_name_or_path": "google/vit-base-patch16-224-in21k",
  
  "architectures": [
  
    "ViTModel"
    
  ],
  
  "attention_probs_dropout_prob": 0.0,
  
  "hidden_act": "gelu",
  
  "hidden_dropout_prob": 0.0,
  
  "hidden_size": 768,
  
  "image_size": 224,
  
  "initializer_range": 0.02,
  
  "intermediate_size": 3072,
  
  "layer_norm_eps": 1e-12,
  
  "model_type": "vit",
  
  "num_attention_heads": 12,
  
  "num_channels": 3,
  
  "num_hidden_layers": 12,
  
  "patch_size": 16,
  
  "qkv_bias": true,
  
  "transformers_version": "4.13.0.dev0"
  
}

Epoch [1/100], Loss: 0.6947, Accuracy: 0.5007

...

Epoch [100/100], Loss: 0.0691, Accuracy: 0.9902

![image](https://github.com/user-attachments/assets/3fb3c525-e6fc-4080-a1c0-c6b0f719be19)

# 1.py

1.py中是不使用预训练模型，自己设计的VIT模型

Epoch [1/100], Loss: 0.7936, Accuracy: 0.5061

Epoch [5/100], Loss: 0.6947, Accuracy: 0.5218

Epoch [10/100], Loss: 0.6914, Accuracy: 0.5321

Epoch [15/100], Loss: 0.6907, Accuracy: 0.5458

Epoch [20/100], Loss: 0.6943, Accuracy: 0.5350

Epoch [25/100], Loss: 0.6897, Accuracy: 0.5409

Epoch [30/100], Loss: 0.6912, Accuracy: 0.5380

Epoch [35/100], Loss: 0.6831, Accuracy: 0.5380

Epoch [40/100], Loss: 0.6722, Accuracy: 0.5737

Epoch [45/100], Loss: 0.6700, Accuracy: 0.5747

Epoch [50/100], Loss: 0.6659, Accuracy: 0.5884

Epoch [55/100], Loss: 0.6672, Accuracy: 0.5532

Epoch [60/100], Loss: 0.6587, Accuracy: 0.5889

Epoch [65/100], Loss: 0.6600, Accuracy: 0.5732

Epoch [70/100], Loss: 0.6360, Accuracy: 0.6144

Epoch [75/100], Loss: 0.6319, Accuracy: 0.6325

Epoch [80/100], Loss: 0.6156, Accuracy: 0.6453

Epoch [85/100], Loss: 0.5991, Accuracy: 0.6688

Epoch [90/100], Loss: 0.5302, Accuracy: 0.7193

Epoch [95/100], Loss: 0.5046, Accuracy: 0.7496

Epoch [100/100], Loss: 0.3853, Accuracy: 0.8256

![image](https://github.com/user-attachments/assets/236c6922-ae70-428e-b3e4-26d11c83113d)

# VGG16+CAM.py

利用CAM机制和VGG网络相结合，在每次迭代后利用CAM机制，重点关注重点位置。

![image](https://github.com/user-attachments/assets/a122db20-b1ef-4b61-a409-717ac377a395)  ![image](https://github.com/user-attachments/assets/f895e633-3c02-4553-ab97-394607582f35)

![image](https://github.com/user-attachments/assets/1143416c-431e-4906-a3ab-63c967a52764)

# MUTI.py

使用VIT作为基本骨干，再使用添加CAM机制的VGG16网络，用于提取更加精细的局部特征并生成显著性区域。

将 ViT 的全局特征与 VGG16（含 CAM）的局部特征融合，采用简单的特征拼接或者注意力加权机制。在融合后的特征上进行分类任务，进一步提升分类精度。

Epoch [1/100] Train Loss: 0.7379, Train Acc: 0.5202 Val Loss: 0.6821, Val Acc: 0.5306

Epoch [5/100] Train Loss: 0.5461, Train Acc: 0.7286 Val Loss: 0.6150, Val Acc: 0.6724

Epoch [10/100] Train Loss: 0.3357, Train Acc: 0.8597 Val Loss: 0.7085, Val Acc: 0.6528

Epoch [15/100] Train Loss: 0.1624, Train Acc: 0.9467 Val Loss: 0.8947, Val Acc: 0.6601

Epoch [20/100] Train Loss: 0.0881, Train Acc: 0.9724 Val Loss: 1.1571, Val Acc: 0.6699

Epoch [25/100] Train Loss: 0.0240, Train Acc: 0.9957 Val Loss: 1.7341, Val Acc: 0.6308

Epoch [30/100] Train Loss: 0.0071, Train Acc: 1.0000 Val Loss: 1.7338, Val Acc: 0.6479

Epoch [35/100] Train Loss: 0.0038, Train Acc: 1.0000 Val Loss: 1.9044, Val Acc: 0.6699

Epoch [40/100] Train Loss: 0.0021, Train Acc: 1.0000 Val Loss: 2.0865, Val Acc: 0.6626

Epoch [45/100] Train Loss: 0.0008, Train Acc: 1.0000 Val Loss: 2.1877, Val Acc: 0.6650

Epoch [50/100] Train Loss: 0.0006, Train Acc: 1.0000 Val Loss: 2.3132, Val Acc: 0.6675

Epoch [55/100] Train Loss: 0.0004, Train Acc: 1.0000 Val Loss: 2.4298, Val Acc: 0.6577

Epoch [60/100] Train Loss: 0.0003, Train Acc: 1.0000 Val Loss: 2.5078, Val Acc: 0.6650

Epoch [65/100] Train Loss: 0.0002, Train Acc: 1.0000 Val Loss: 2.6108, Val Acc: 0.6626

Epoch [70/100] Train Loss: 0.0002, Train Acc: 1.0000 Val Loss: 2.6989, Val Acc: 0.6626

Epoch [75/100] Train Loss: 0.0002, Train Acc: 1.0000 Val Loss: 2.7836, Val Acc: 0.6650

Epoch [80/100] Train Loss: 0.0001, Train Acc: 1.0000 Val Loss: 2.8511, Val Acc: 0.6675

Epoch [85/100] Train Loss: 0.0001, Train Acc: 1.0000 Val Loss: 2.9525, Val Acc: 0.6601

Epoch [90/100] Train Loss: 0.0001, Train Acc: 1.0000 Val Loss: 3.0051, Val Acc: 0.6626

Epoch [95/100] Train Loss: 0.0001, Train Acc: 1.0000 Val Loss: 3.0521, Val Acc: 0.6650

Epoch [100/100] Train Loss: 0.0001, Train Acc: 1.0000 Val Loss: 3.1835, Val Acc: 0.6577

![image](https://github.com/user-attachments/assets/466aa5a9-ffd7-4bd9-90e1-7df1274a044f)
