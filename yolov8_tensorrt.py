import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os
import time
from ultralytics import YOLO

# 配置日志
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

class YOLOv8TensorRT:
    def __init__(self, engine_path, input_shape=(640, 640), confidence_threshold=0.25, iou_threshold=0.45):
        """初始化YOLOv8 TensorRT推理器"""
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.input_shape = input_shape
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # 获取输入输出信息
        self.input_tensor_name = None
        self.output_tensor_names = []
        
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.input_tensor_name = tensor_name
                self.input_dims = self.engine.get_tensor_shape(tensor_name)
            else:
                self.output_tensor_names.append(tensor_name)
        
        # 分配缓冲区
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        
        # 类别名称（COCO数据集）
        self.class_names = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
            "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
            "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]
    
    def load_engine(self, engine_path):
        """加载TensorRT引擎"""
        runtime = trt.Runtime(TRT_LOGGER)
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        engine = runtime.deserialize_cuda_engine(engine_data)
        return engine
    
    def allocate_buffers(self):
        """分配输入输出缓冲区"""
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for i in range(self.engine.num_io_tensors):
            tensor_name = engine.get_tensor_name(i)
            dims = self.engine.get_tensor_shape(tensor_name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            
            # 计算缓冲区大小
            size = trt.volume(dims)
            host_mem = cuda.pagelocked_empty(size, dtype)
            
            # 分配设备内存
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            # 添加到绑定列表
            bindings.append(int(device_mem))
            
            # 区分输入和输出
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append({
                    "host": host_mem,
                    "device": device_mem,
                    "name": tensor_name,
                    "dims": dims
                })
            else:
                outputs.append({
                    "host": host_mem,
                    "device": device_mem,
                    "name": tensor_name,
                    "dims": dims
                })
        
        return inputs, outputs, bindings, stream
    
    def preprocess(self, image):
        """预处理输入图像"""
        # 保存原始图像尺寸
        original_height, original_width = image.shape[:2]
        
        # 调整图像大小并保持纵横比
        input_width, input_height = self.input_shape
        scale = min(input_width / original_width, input_height / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # 调整大小
        resized_image = cv2.resize(image, (new_width, new_height))
        
        # 创建带填充的输入图像
        input_image = np.zeros((input_height, input_width, 3), dtype=np.uint8)
        offset_x = (input_width - new_width) // 2
        offset_y = (input_height - new_height) // 2
        input_image[offset_y:offset_y+new_height, offset_x:offset_x+new_width, :] = resized_image
        
        # 转换为RGB格式并归一化
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = input_image.astype(np.float32) / 255.0
        
        # 调整维度 (HWC -> CHW)
        input_image = input_image.transpose(2, 0, 1)
        
        # 添加批次维度
        input_image = np.expand_dims(input_image, axis=0)
        
        return input_image, scale, offset_x, offset_y
    
    def postprocess(self, outputs, scale, offset_x, offset_y, original_width, original_height):
        """后处理输出结果"""
        # 获取输出数据
        output_data = outputs[0]["host"].reshape(outputs[0]["dims"])
        
        # 解析检测结果
        detections = []
        for row in output_data[0]:
            confidence = row[4]
            if confidence < self.confidence_threshold:
                continue
                
            # 获取类别分数和类别ID
            class_scores = row[5:]
            class_id = np.argmax(class_scores)
            class_score = class_scores[class_id]
            
            # 计算置信度
            confidence = confidence * class_score
            if confidence < self.confidence_threshold:
                continue
                
            # 获取边界框坐标
            x_center = row[0]
            y_center = row[1]
            width = row[2]
            height = row[3]
            
            # 将坐标从输入尺寸转换回原始图像尺寸
            x_center = (x_center - offset_x) / scale
            y_center = (y_center - offset_y) / scale
            width /= scale
            height /= scale
            
            # 计算左上角和右下角坐标
            x1 = max(0, int(x_center - width / 2))
            y1 = max(0, int(y_center - height / 2))
            x2 = min(original_width, int(x_center + width / 2))
            y2 = min(original_height, int(y_center + height / 2))
            
            detections.append({
                "class_id": class_id,
                "class_name": self.class_names[class_id],
                "confidence": float(confidence),
                "bbox": [x1, y1, x2, y2]
            })
        
        # 应用非极大值抑制
        if not detections:
            return []
            
        # 按置信度排序
        detections.sort(key=lambda x: x["confidence"], reverse=True)
        keep = []
        
        while detections:
            current = detections.pop(0)
            keep.append(current)
            
            # 计算与剩余检测框的IoU并过滤
            new_detections = []
            for detection in detections:
                iou = self.calculate_iou(current["bbox"], detection["bbox"])
                if iou < self.iou_threshold:
                    new_detections.append(detection)
            detections = new_detections
            
        return keep
    
    def calculate_iou(self, bbox1, bbox2):
        """计算两个边界框的交并比(IoU)"""
        x1, y1, x2, y2 = bbox1
        x1_, y1_, x2_, y2_ = bbox2
        
        # 计算交集区域
        inter_x1 = max(x1, x1_)
        inter_y1 = max(y1, y1_)
        inter_x2 = min(x2, x2_)
        inter_y2 = min(y2, y2_)
        
        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            return 0.0
            
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        
        # 计算并集区域
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2_ - x1_) * (y2_ - y1_)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area
    
    def infer(self, image):
        """执行推理"""
        # 预处理
        original_height, original_width = image.shape[:2]
        input_data, scale, offset_x, offset_y = self.preprocess(image)
        
        # 设置输入形状
        self.context.set_input_shape(self.input_tensor_name, input_data.shape)
        
        # 复制输入数据到设备
        np.copyto(self.inputs[0]["host"], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)
        
        # 执行推理
        start_time = time.time()
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        
        # 复制输出数据到主机
        for output in self.outputs:
            cuda.memcpy_dtoh_async(output["host"], output["device"], self.stream)
        
        self.stream.synchronize()
        inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
        
        # 后处理
        results = self.postprocess(self.outputs, scale, offset_x, offset_y, original_width, original_height)
        
        return results, inference_time

def export_yolov8_to_onnx(model_name="yolov8s.pt", opset=12, dynamic=False, simplify=True):
    """导出YOLOv8模型为ONNX格式"""
    onnx_path = model_name.replace(".pt", ".onnx")
    
    if not os.path.exists(onnx_path):
        print(f"导出 {model_name} 到 ONNX...")
        model = YOLO(model_name)
        success = model.export(
            format="onnx",
            opset=opset,
            dynamic=dynamic,
            simplify=simplify,
            half=True  # 启用FP16精度
        )
        
        if not success:
            raise Exception("ONNX导出失败")
    
    return onnx_path

def build_yolov8_engine(onnx_path, engine_path="yolov8_engine.trt", precision="fp16", workspace_size=2048):
    """从ONNX模型构建TensorRT引擎"""
    if not os.path.exists(engine_path):
        print(f"构建 {precision} 精度引擎...")
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        # 解析ONNX模型
        with open(onnx_path, "rb") as model_file:
            if not parser.parse(model_file.read()):
                print("ERROR: 解析ONNX文件失败")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        # 配置构建器
        config = builder.create_builder_config()
        config.max_workspace_size = workspace_size * 1024 * 1024  # 转换为字节
        
        # 设置精度模式
        if precision == "fp16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == "int8" and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            # 注意：INT8量化需要校准器，此处省略实现
        
        # 构建并保存引擎
        serialized_engine = builder.build_serialized_network(network, config)
        if not serialized_engine:
            raise Exception("引擎构建失败")
            
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)
        
        print(f"引擎已保存至 {engine_path}")
    
    return engine_path

def main():
    # 配置参数
    model_name = "yolov8s.pt"
    engine_path = "yolov8s_engine.trt"
    precision = "fp16"
    confidence_threshold = 0.4
    iou_threshold = 0.5
    input_shape = (640, 640)
    
    # 1. 导出ONNX模型
    onnx_path = export_yolov8_to_onnx(model_name, opset=12)
    
    # 2. 构建TensorRT引擎
    build_yolov8_engine(onnx_path, engine_path, precision)
    
    # 3. 初始化YOLOv8 TensorRT推理器
    yolov8_trt = YOLOv8TensorRT(
        engine_path,
        input_shape=input_shape,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold
    )
    
    # 4. 打开摄像头
    cap = cv2.VideoCapture(0)  # 使用默认摄像头
    
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # 用于计算FPS
    fps_counter = 0
    fps_start_time = time.time()
    
    print("按 'q' 键退出...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取帧")
            break
        
        # 执行推理
        results, inference_time = yolov8_trt.infer(frame)
        
        # 绘制检测结果
        for result in results:
            x1, y1, x2, y2 = result["bbox"]
            class_name = result["class_name"]
            confidence = result["confidence"]
            
            # 绘制边界框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制类别名称和置信度
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # 显示推理时间和FPS
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()
        else:
            fps = 0
        
        # 显示性能信息
        cv2.putText(frame, f"Inference Time: {inference_time:.2f} ms", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        # 显示结果
        cv2.imshow("YOLOv8 TensorRT Object Detection", frame)
        
        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
