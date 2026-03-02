import tensorflow as tf
import numpy as np
import os

# ================= 配置区域 =================
# 1. 输入：你本地的 Float32 TFLite 模型路径
input_model_path = "model_static.tflite" 

# 2. 输出：生成的 Int8 全量化模型路径
output_model_path = "model_static_int8.tflite"

# 3. 输入形状：需要知道模型的输入形状 (例如 [1, 224, 224, 3])
# 如果不确定，脚本后面会自动检测并打印，你可以先运行一次看看
input_shape = (1, 3, 224, 224) 
# ===========================================

def representative_dataset():
    """
    代表性数据集生成器。
    用于校准量化参数。
    注意：这里使用随机噪声作为演示。
    为了获得高精度，请替换为真实的图片数据 (归一化到 0.0-1.0 之间)。
    """
    print("正在生成代表性数据进行校准...")
    # 校准通常需要 100-500 张图片
    num_calibration_steps = 100 
    
    for _ in range(num_calibration_steps):
        # 生成随机浮点数数据 [0.0, 1.0]
        # 形状必须与模型输入一致
        data = np.random.rand(*input_shape).astype(np.float32)
        
        # 如果模型输入有多个，yield 一个列表，例如: yield [data1, data2]
        # 大多数图像模型只有一个输入
        yield [data]

def convert():
    if not os.path.exists(input_model_path):
        print(f"错误：找不到文件 {input_model_path}")
        return

    print(f"正在加载模型：{input_model_path} ...")
    
    # 1. 从本地文件加载转换器
    converter = tf.lite.TFLiteConverter.from_model_file(input_model_path)

    # 2. 【关键】启用默认优化 (触发量化)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # 3. 【关键】绑定代表性数据集
    # 这一步会运行推理来收集激活值的分布范围，从而计算 scale 和 zero_point
    converter.representative_dataset = representative_dataset

    # 4. 【关键】强制使用 INT8 算子集
    # 这会确保所有支持的算子都转换为 int8，不支持的会报错或保留 float (如果不加下面两行限制)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    
    # 5. 【关键】保持输入输出为 float32 (可选但推荐)
    # 这样生成的模型外部接口依然是 float32，内部自动插入 Quantize/Dequantize
    # 这完全符合你之前展示的 ValidateNode 代码逻辑
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    # 6. (可选) 如果某些算子不支持 int8，你想让它们回退到 float，可以注释掉上面那行 supported_ops
    # 但为了适配你的 NPU Delegate，建议保持上面的设置，让它严格检查。

    print("开始转换 (这可能需要几分钟)...")
    try:
        tflite_model_quant = converter.convert()
        
        with open(output_model_path, 'wb') as f:
            f.write(tflite_model_quant)
            
        print(f"✅ 成功！全整型量化模型已保存至：{output_model_path}")
        
        # 简单的大小对比
        original_size = os.path.getsize(input_model_path) / 1024 / 1024
        new_size = os.path.getsize(output_model_path) / 1024 / 1024
        print(f"   原始大小：{original_size:.2f} MB")
        print(f"   量化大小：{new_size:.2f} MB (通常变小)")
        
    except Exception as e:
        print(f"❌ 转换失败：{e}")
        print("提示：如果报错说某些算子不支持 int8，说明该算子在 NPU 上可能也不支持，或者需要更新 TF 版本。")

if __name__ == "__main__":
    # 自动检测输入形状 (如果用户没改的话)
    # 这是一个辅助功能，防止用户填错形状
    if input_shape == (1, 3, 224, 224):
        try:
            interpreter = tf.lite.Interpreter(model_path=input_model_path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            detected_shape = tuple(input_details[0]['shape'])
            print(f"检测到模型输入形状为：{detected_shape}")
            print("如果上述形状与你实际数据不符，请修改脚本中的 input_shape 变量。")
            input_shape = detected_shape
        except Exception:
            print("无法自动检测形状，将使用默认值 (1, 224, 224, 3)。如果出错请手动修改。")

    convert()
