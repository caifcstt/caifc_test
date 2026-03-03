import tensorflow as tf
import numpy as np
import os
import json

# 【关键修改 1】导入 keras_hub
# 这行代码会自动注册 ResNetBackbone 等自定义类到 Keras 全局注册表中
try:
    import keras_hub
    print("✅ keras_hub 导入成功，自定义层已注册。")
except ImportError:
    print("❌ 错误：未找到 keras_hub 库。请先运行：pip install keras-hub")
    exit(1)

# ================= 配置区域 =================
config_file = "config.json"
main_weights = "model.weights.h5"
task_weights = "task.weights.h5"
preprocessor_file = "preprocessor.json"
output_model_path = "model_int8_224.tflite"
NUM_CALIBRATION_STEPS = 100
# ===========================================

def get_input_config():
    default_shape = (1, 224, 224, 3)
    default_mean = [0.0, 0.0, 0.0]
    default_std = [1.0, 1.0, 1.0]
    
    if not os.path.exists(preprocessor_file):
        print(f"⚠️ 未找到 {preprocessor_file}，使用默认配置: Shape={default_shape}")
        return default_shape, default_mean, default_std
    
    try:
        with open(preprocessor_file, 'r') as f:
            data = json.load(f)
        
        shape = default_shape
        if "input_shape" in data:
            s = data["input_shape"]
            if isinstance(s, list):
                if len(s) == 3:
                    shape = (1,) + tuple(s)
                elif len(s) == 4:
                    shape = tuple(s)
        elif "image_size" in data:
            s = data["image_size"]
            if isinstance(s, int):
                shape = (1, s, s, 3)
            elif isinstance(s, list) and len(s) == 2:
                shape = (1, s[0], s[1], 3)
        
        mean = data.get("mean", default_mean)
        std = data.get("std", default_std)
        
        print(f"✅ 从 {preprocessor_file} 读取配置: Shape={shape}")
        return shape, mean, std
        
    except Exception as e:
        print(f"⚠️ 解析 {preprocessor_file} 失败: {e}，使用默认值。")
        return default_shape, default_mean, default_std

def representative_dataset(input_shape, mean, std):
    print(f"正在生成 {NUM_CALIBRATION_STEPS} 组校准数据 (Shape: {input_shape})...")
    for i in range(NUM_CALIBRATION_STEPS):
        data = np.random.rand(*input_shape).astype(np.float32)
        yield [data]

def convert():
    required_files = [config_file, main_weights]
    for f in required_files:
        if not os.path.exists(f):
            print(f"❌ 错误：找不到关键文件 {f}")
            return

    input_shape, mean, std = get_input_config()

    print("\n1. 正在从 config.json 重建模型结构...")
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config_json = f.read()
        
        # 【关键修改 2】加载模型
        # 现在因为导入了 keras_hub，model_from_json 能识别 'keras_hub>ResNetBackbone' 了
        # 为了双重保险，也可以显式传递 custom_objects，但通常导入库就足够了
        model = tf.keras.models.model_from_json(config_json)
        print("   ✅ 模型结构构建完成 (包含 Keras Hub 自定义层)")
        
        print("\n2. 正在加载权重...")
        
        # 加载主权重
        try:
            model.load_weights(main_weights, skip_mismatch=False)
            print(f"   ✅ 成功加载主权重：{main_weights}")
        except Exception as e:
            print(f"   ⚠️ 加载主权重警告：{str(e)[:60]}... 尝试跳过不匹配项")
            model.load_weights(main_weights, skip_mismatch=True)

        # 加载任务权重
        if os.path.exists(task_weights):
            try:
                model.load_weights(task_weights, skip_mismatch=True)
                print(f"   ✅ 成功补充加载任务权重：{task_weights}")
            except Exception as e:
                print(f"   ⚠️ 加载任务权重失败：{str(e)[:60]}")
        
        # 验证推理
        dummy = tf.random.normal(input_shape)
        _ = model(dummy, training=False)
        print("   ✅ 模型推理测试通过！")

    except Exception as e:
        print(f"\n❌ 模型重建或加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n3. 初始化 TFLite 转换器...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    print("4. 配置 INT8 量化参数...")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(input_shape, mean, std)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    converter._experimental_lower_tensor_list_ops = False

    print("5. 开始转换 (这可能需要几分钟)...")
    try:
        tflite_model_quant = converter.convert()
        
        with open(output_model_path, 'wb') as f:
            f.write(tflite_model_quant)
            
        new_size = os.path.getsize(output_model_path) / 1024 / 1024
        print(f"\n🎉 成功！INT8 量化模型已保存：{output_model_path}")
        print(f"   文件大小：{new_size:.2f} MB")
        
    except Exception as e:
        print(f"\n❌ 转换失败：{e}")
        # 提供混合精度回退建议
        print("\n💡 如果是因为算子不支持 INT8，请尝试在脚本中启用 SELECT_TF_OPS。")

if __name__ == "__main__":
    convert()
