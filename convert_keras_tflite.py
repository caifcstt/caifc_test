import tensorflow as tf
import numpy as np
import os

# ================= 配置区域 =================
# 1. 输入：原始模型路径 (.h5 或 SavedModel 文件夹)
# 如果是 .h5 文件: "model.h5"
# 如果是 SavedModel 文件夹: "saved_model_directory"
input_model_path = "mobilenet_v2.h5" 

# 2. 输出：生成的 Int8 全量化模型路径
output_model_path = "mobilenet_v2_int8.tflite"

# 3. 校准数据数量 (用于代表性数据集)
# 通常 100-500 张图片即可，越多越准但越慢
num_calibration_steps = 100
# ===========================================

def representative_dataset():
    """
    代表性数据集生成器。
    这是量化的核心：提供真实数据让 TF 计算激活值的范围 (Scale/Zero-point)。
    """
    print(f"正在生成 {num_calibration_steps} 组代表性数据进行校准...")
    print("⚠️ 注意：当前使用随机噪声。为了高精度，请替换为真实图片代码（见下方注释）。")

    for _ in range(num_calibration_steps):
        # ---------------------------------------------------------
        # 【方案 A】使用随机噪声 (快速测试，精度较差)
        # 假设输入形状是 [1, 224, 224, 3]，请根据实际模型修改
        # 如果你的模型输入是 NHWC (Batch, H, W, C)，通常是 (1, 224, 224, 3)
        # 如果你的模型输入是 NCHW (Batch, C, H, W)，通常是 (1, 3, 224, 224)
        # 这里先尝试自动检测或默认 NHWC
        data = np.random.rand(1, 224, 224, 3).astype(np.float32)
        
        # 如果模型需要归一化到 -1~1 或 0~1，随机数已经是 0~1，通常够用
        # 如果需要特定均值方差，可以在这里处理
        
        yield [data]
        # ---------------------------------------------------------
        
        # ---------------------------------------------------------
        # 【方案 B】使用真实图片 (推荐！精度高)
        # 如果你有 ImageNet 验证集或其他图片，取消下面注释并修改路径
        # ---------------------------------------------------------
        # img_path = f"/path/to/images/image_{_}.jpg"
        # img = tf.io.read_file(img_path)
        # img = tf.image.decode_jpeg(img, channels=3)
        # img = tf.image.resize(img, [224, 224]) # 调整到模型输入大小
        # img = tf.cast(img, tf.float32) / 255.0 # 归一化
        # yield [tf.expand_dims(img, axis=0)]

def convert():
    if not os.path.exists(input_model_path):
        print(f"❌ 错误：找不到文件 {input_model_path}")
        print("请确认路径是否正确，或者是否是一个文件夹 (SavedModel)。")
        return

    print(f"正在加载原始模型：{input_model_path} ...")
    
    try:
        # 1. 加载模型
        # 自动判断是 .h5 还是 SavedModel 文件夹
        if input_model_path.endswith('.h5'):
            model = tf.keras.models.load_model(input_model_path)
            print("检测到 Keras .h5 模型。")
        else:
            # 假设是 SavedModel 文件夹
            model = tf.keras.models.load_model(input_model_path)
            print("检测到 SavedModel 格式。")
            
        # 打印模型摘要，确认输入形状
        model.summary()
        
        # 获取输入形状以便生成正确的随机数据
        input_shape = model.input_shape
        print(f"模型输入形状检测到：{input_shape}")
        
        # 动态更新 representative_dataset 中的形状 (简单的 Hack)
        # 注意：上面的 generator 是硬编码的，这里为了简单演示，
        # 严谨的做法是将 input_shape 作为参数传给 generator 类
        # 这里我们假设用户会手动修改 generator 里的形状，或者模型是标准的 NHWC
        
    except Exception as e:
        print(f"加载模型失败：{e}")
        return

    # 2. 创建转换器
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # 3. 【关键】启用默认优化 (触发量化)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # 4. 【关键】绑定代表性数据集
    # ⚠️ 重要：这里的 generator 必须产生与模型输入形状一致的数据
    # 如果上面的 input_shape 是 (None, 3, 224, 224) (NCHW)，你需要修改 generator
    converter.representative_dataset = representative_dataset

    # 5. 【关键】强制使用 INT8 算子集
    # 这确保生成的模型内部全是 int8 运算
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # 6. 【关键】保持输入输出为 float32
    # 这样模型外部接口友好，内部自动插入 Quantize/Dequantize
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    print("开始转换并量化 (这可能需要几分钟)...")
    try:
        tflite_model_quant = converter.convert()
        
        with open(output_model_path, 'wb') as f:
            f.write(tflite_model_quant)
            
        print(f"\n✅ 成功！全整型量化模型已保存至：{output_model_path}")
        
        original_size = os.path.getsize(input_model_path) / 1024 / 1024 if os.path.isfile(input_model_path) else 0
        new_size = os.path.getsize(output_model_path) / 1024 / 1024
        print(f"   量化后大小：{new_size:.2f} MB")
        print(f"   (原始模型大小仅供参考，量化主要减小的是权重体积)")
        
        # 验证生成的模型
        print("\n正在验证生成的模型...")
        interpreter = tf.lite.Interpreter(model_path=output_model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"输入节点类型：{input_details[0]['dtype']} (应为 float32)")
        print(f"输出节点类型：{output_details[0]['dtype']} (应为 float32)")
        
        # 检查第一个算子是否是 QUANTIZE
        details = interpreter.get_tensor_details()
        # 简单检查：如果模型正确量化，中间张量应该是 int8
        int8_count = sum(1 for d in details if d['dtype'] == np.int8)
        print(f"模型中 int8 张量数量：{int8_count} (如果 > 0 说明量化成功)")
        
    except Exception as e:
        print(f"\n❌ 转换失败：{e}")
        print("\n常见原因:")
        print("1. 代表性数据集形状与模型输入不匹配。请修改 representative_dataset 函数。")
        print("2. 模型中包含不支持 INT8 的算子。尝试移除 target_spec 限制看报错信息。")
        print("3. TensorFlow 版本问题。建议尝试 TF 2.13 - 2.15。")

if __name__ == "__main__":
    convert()
