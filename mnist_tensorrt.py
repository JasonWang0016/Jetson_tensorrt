import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os
from PIL import Image
import time

# 配置日志
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

class MNISTCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_images, batch_size=8, input_shape=(1, 28, 28)):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.input_size = trt.volume(input_shape) * batch_size
        self.cache_file = "mnist_calibration.cache"
        
        # 准备校准数据集
        self.images = []
        for img_path in calibration_images[:1024]:  # 使用前1024张图像
            img = Image.open(img_path).convert('L')
            img = img.resize((input_shape[1], input_shape[2]))
            img = np.array(img, dtype=np.float32) / 255.0
            img = img[np.newaxis, :, :]  # 添加通道维度
            self.images.append(img)
        
        self.batch_idx = 0
        self.max_batches = len(self.images) // batch_size
        
        # 分配设备内存
        self.device_input = cuda.mem_alloc(self.input_size * 4)  # FP32
        
    def get_batch_size(self):
        return self.batch_size
    
    def get_batch(self, names):
        if self.batch_idx >= self.max_batches:
            return None
        
        # 准备批次数据
        batch = self.images[self.batch_idx * self.batch_size : (self.batch_idx + 1) * self.batch_size]
        batch = np.concatenate(batch).ravel()
        
        # 复制到设备
        cuda.memcpy_htod(self.device_input, batch)
        self.batch_idx += 1
        return [int(self.device_input)]
    
    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None
    
    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

def build_engine(onnx_file_path, precision="fp32", calibrator=None):
    """从ONNX模型构建TensorRT引擎"""
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # 解析ONNX模型
    with open(onnx_file_path, "rb") as model_file:
        if not parser.parse(model_file.read()):
            print("ERROR: Failed to parse the ONNX file.")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # 配置构建器
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB工作空间
    
    # 设置精度模式
    if precision == "fp16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8" and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        if not calibrator:
            raise ValueError("需要校准器进行INT8量化")
        config.int8_calibrator = calibrator
    
    # 构建并序列化引擎
    serialized_engine = builder.build_serialized_network(network, config)
    return serialized_engine

def load_engine(engine_path):
    """加载序列化的TensorRT引擎"""
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    engine = runtime.deserialize_cuda_engine(engine_data)
    return engine

def preprocess_image(image_path, input_shape):
    """预处理输入图像"""
    img = Image.open(image_path).convert('L')  # 转为灰度图
    img = img.resize((input_shape[1], input_shape[2]))
    img = np.array(img, dtype=np.float32) / 255.0  # 归一化到[0,1]
    img = img[np.newaxis, np.newaxis, :, :]  # 添加批次和通道维度
    return img

def postprocess_output(output_data, input_shape):
    """后处理输出结果"""
    # 将输出转换为概率分布
    probabilities = np.exp(output_data) / np.sum(np.exp(output_data))
    predicted_class = np.argmax(probabilities)
    confidence = probabilities[predicted_class]
    
    return {
        "class_id": int(predicted_class),
        "confidence": float(confidence),
        "probabilities": probabilities.tolist()
    }

def main():
    # 配置参数
    onnx_model_path = "mnist.onnx"
    engine_path = "mnist_engine.trt"
    test_image_path = "test_digit.png"
    input_shape = (1, 28, 28)  # (通道, 高度, 宽度)
    precision = "fp16"  # 可选: fp32, fp16, int8
    
    # 1. 构建或加载引擎
    if not os.path.exists(engine_path):
        print(f"构建{precision}精度引擎...")
        if precision == "int8":
            # 准备校准数据集（这里使用MNIST测试集的前1000张图像）
            calibration_images = [f"mnist_data/test/{i}.png" for i in range(1000)]
            calibrator = MNISTCalibrator(calibration_images, batch_size=8, input_shape=input_shape)
            engine_data = build_engine(onnx_model_path, precision, calibrator)
        else:
            engine_data = build_engine(onnx_model_path, precision)
        
        with open(engine_path, "wb") as f:
            f.write(engine_data)
        print(f"引擎已保存至 {engine_path}")
    else:
        print(f"加载现有引擎 {engine_path}")
        engine = load_engine(engine_path)
    
    # 2. 准备输入数据
    input_data = preprocess_image(test_image_path, input_shape)
    
    # 3. 执行推理
    context = engine.create_execution_context()
    
    # 分配缓冲区
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    
    # 为每个张量分配缓冲区
    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        dims = engine.get_tensor_shape(tensor_name)
        dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))
        
        # 计算缓冲区大小
        size = trt.volume(dims)
        host_mem = cuda.pagelocked_empty(size, dtype)
        
        # 分配设备内存
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        
        # 添加到绑定列表
        bindings.append(int(device_mem))
        
        # 区分输入和输出
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            inputs.append({
                "host": host_mem,
                "device": device_mem,
                "name": tensor_name
            })
            # 设置输入形状
            context.set_input_shape(tensor_name, dims)
        else:
            outputs.append({
                "host": host_mem,
                "device": device_mem,
                "name": tensor_name
            })
    
    # 复制输入数据到设备
    np.copyto(inputs[0]["host"], input_data.ravel())
    cuda.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)
    
    # 执行推理并计时
    start_time = time.time()
    context.execute_async_v3(stream_handle=stream.handle)
    stream.synchronize()
    inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
    
    # 复制输出数据到主机
    cuda.memcpy_dtoh_async(outputs[0]["host"], outputs[0]["device"], stream)
    stream.synchronize()
    
    # 4. 后处理结果
    result = postprocess_output(outputs[0]["host"], input_shape)
    
    # 5. 显示结果
    print(f"\n推理结果:")
    print(f"预测数字: {result['class_id']}")
    print(f"置信度: {result['confidence']:.4f}")
    print(f"推理时间: {inference_time:.2f} ms")
    
    # 6. 性能评估（运行100次推理）
    print("\n性能评估:")
    start_time = time.time()
    for _ in range(100):
        cuda.memcpy_htod_async(inputs[0]["device"], inputs[0]["host"], stream)
        context.execute_async_v3(stream_handle=stream.handle)
        stream.synchronize()
    avg_time = (time.time() - start_time) * 1000 / 100  # 平均时间(毫秒)
    print(f"平均推理时间: {avg_time:.2f} ms")
    print(f"吞吐量: {1000/avg_time:.2f} FPS")

if __name__ == "__main__":
    main()
