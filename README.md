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

