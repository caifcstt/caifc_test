import tensorflow as tf

print("正在下载 MobileNet V2 ...")
# alpha=1.0 是标准版，0.35 是超轻量版
model = tf.keras.applications.MobileNetV2(
    weights='imagenet', 
    input_shape=(224, 224, 3), 
    include_top=True
)

print("保存为 mobilenet_v2.h5 ...")
model.save('mobilenet_v2.h5')
print("✅ 完成！")
