import tensorflow as tf

print("正在下载 ResNet-50 模型 (约 100MB)...")
# 从官方源下载预训练权重的 ResNet50
model = tf.keras.applications.ResNet50(
    weights='imagenet', 
    input_shape=(224, 224, 3), 
    include_top=True # 包含全连接层，用于分类
)

print("下载完成，正在保存为 resnet50.h5 ...")
model.save('resnet50.h5')

print("✅ 成功！文件已保存至当前目录：resnet50.h5")
print("现在你可以运行量化脚本了。")
