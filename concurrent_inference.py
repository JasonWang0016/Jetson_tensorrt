import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os
import time
import threading
from queue import Queue

# 配置日志
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

class TensorRTEngine:
    """TensorRT引擎封装类"""
    def __init__(self, engine_path):
        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.input_tensor_name = None
        
        # 获取输入张量名称
        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                self.input_tensor_name = tensor_name
                break
    
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
            tensor_name = self.engine.get_tensor_name(i)
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
    
    def infer(self, input_data):
        """执行推理"""
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
        
        # 整理输出结果
        results = {}
        for output in self.outputs:
            results[output["name"]] = output["host"].reshape(output["dims"])
        
        return results, inference_time

class YOLOv8Detector(TensorRTEngine):
    """YOLOv8目标检测引擎"""
    def __init__(self, engine_path, input_shape=(640, 640), confidence_threshold=0.25, iou_threshold=0.45):
        super().__init__(engine_path)
        self.input_shape = input_shape
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # COCO数据集类别名称
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
        
        return input_image, scale, offset_x, offset_y, original_width, original_height
    
    def postprocess(self, outputs, scale, offset_x, offset_y, original_width, original_height):
        """后处理输出结果"""
        # 获取输出数据 (假设只有一个输出张量)
        output_name = next(iter(outputs.keys()))
        output_data = outputs[output_name].reshape(self.outputs[0]["dims"])
        
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
    
    def process_frame(self, frame):
        """处理单帧图像"""
        input_data, scale, offset_x, offset_y, original_width, original_height = self.preprocess(frame)
        outputs, inference_time = self.infer(input_data)
        results = self.postprocess(outputs, scale, offset_x, offset_y, original_width, original_height)
        return results, inference_time

class PoseEstimator(TensorRTEngine):
    """姿态估计引擎"""
    def __init__(self, engine_path, input_shape=(256, 192), confidence_threshold=0.5):
        super().__init__(engine_path)
        self.input_shape = input_shape
        self.confidence_threshold = confidence_threshold
        
        # COCO关键点名称
        self.keypoint_names = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]
        
        # 关键点连接方式
        self.skeleton = [
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (5, 17),
            (6, 17), (17, 0), (0, 1), (1, 3), (0, 2), (2, 4)
        ]
        
        # 关键点颜色
        self.keypoint_colors = [
            (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0),
            (85, 255, 0), (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255),
            (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255), (170, 0, 255),
            (255, 0, 255), (255, 0, 170), (255, 0, 85)
        ]
    
    def preprocess(self, image, bbox=None):
        """预处理输入图像"""
        # 如果提供了边界框，则裁剪图像
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            cropped_image = image[y1:y2, x1:x2]
        else:
            cropped_image = image.copy()
            bbox = [0, 0, image.shape[1], image.shape[0]]
        
        # 保存原始图像尺寸
        original_height, original_width = cropped_image.shape[:2]
        
        # 调整图像大小
        input_height, input_width = self.input_shape
        resized_image = cv2.resize(cropped_image, (input_width, input_height))
        
        # 转换为RGB格式并归一化
        input_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        input_image = input_image.astype(np.float32) / 255.0
        
        # 调整维度 (HWC -> CHW)
        input_image = input_image.transpose(2, 0, 1)
        
        # 添加批次维度
        input_image = np.expand_dims(input_image, axis=0)
        
        return input_image, original_width, original_height, bbox
    
    def postprocess(self, outputs, original_width, original_height, bbox):
        """后处理输出结果"""
        # 获取输出数据 (假设只有一个输出张量)
        output_name = next(iter(outputs.keys()))
        output_data = outputs[output_name]
        
        # 解析关键点
        keypoints = []
        if len(output_data.shape) == 4:  # (batch, channels, height, width)
            # 简单姿态估计模型输出解析
            heatmaps = output_data[0]
            height, width = heatmaps.shape[1], heatmaps.shape[2]
            
            for i in range(heatmaps.shape[0]):
                heatmap = heatmaps[i]
                confidence = np.max(heatmap)
                
                if confidence < self.confidence_threshold:
                    keypoints.append(None)
                    continue
                    
                # 找到关键点位置
                y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                
                # 转换回原始图像坐标
                x = int(x / width * original_width) + bbox[0]
                y = int(y / height * original_height) + bbox[1]
                
                keypoints.append({
                    "x": x,
                    "y": y,
                    "confidence": float(confidence),
                    "name": self.keypoint_names[i] if i < len(self.keypoint_names) else f"keypoint_{i}"
                })
        else:
            # 更复杂的姿态估计模型输出解析 (此处简化处理)
            pass
            
        return {"keypoints": keypoints}
    
    def process_detection(self, frame, detection):
        """处理单个检测结果的姿态估计"""
        x1, y1, x2, y2 = detection["bbox"]
        bbox = [x1, y1, x2, y2]
        
        # 预处理
        input_data, original_width, original_height, _ = self.preprocess(frame, bbox)
        
        # 推理
        outputs, inference_time = self.infer(input_data)
        
        # 后处理
        results = self.postprocess(outputs, original_width, original_height, bbox)
        
        return results, inference_time
    
    def draw_poses(self, frame, pose_results):
        """绘制姿态估计结果"""
        # 绘制关键点
        for keypoint in pose_results["keypoints"]:
            if keypoint:
                x, y = keypoint["x"], keypoint["y"]
                cv2.circle(frame, (x, y), 5, self.keypoint_colors[self.keypoint_names.index(keypoint["name"])], -1)
        
        # 绘制骨架连接
        for (i, j) in self.skeleton:
            if i < len(pose_results["keypoints"]) and j < len(pose_results["keypoints"]):
                keypoint_i = pose_results["keypoints"][i]
                keypoint_j = pose_results["keypoints"][j]
                
                if keypoint_i and keypoint_j:
                    x1, y1 = keypoint_i["x"], keypoint_i["y"]
                    x2, y2 = keypoint_j["x"], keypoint_j["y"]
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return frame

def model_inference_thread(model, input_queue, output_queue, stop_event):
    """模型推理线程"""
    while not stop_event.is_set():
        if not input_queue.empty():
            # 获取输入数据
            input_data = input_queue.get()
            
            # 执行推理
            if isinstance(model, YOLOv8Detector):
                results, inference_time = model.process_frame(input_data["frame"])
            elif isinstance(model, PoseEstimator) and "detection" in input_data:
                results, inference_time = model.process_detection(input_data["frame"], input_data["detection"]["bbox"])
            else:
                results = None
                inference_time = 0
                
            # 将结果放入输出队列
            output_queue.put({
                "results": results,
                "inference_time": inference_time,
                "timestamp": input_data["timestamp"]
            })
            
            input_queue.task_done()
        else:
            time.sleep(0.001)  # 短暂休眠，减少CPU占用

def main():
    """多模型并发推理主函数"""
    # 配置参数
    yolov8_engine_path = "yolov8s_engine.trt"
    pose_engine_path = "pose_engine.trt"
    confidence_threshold = 0.4
    iou_threshold = 0.5
    
    # 初始化模型引擎
    print("加载目标检测引擎...")
    detector = YOLOv8Detector(
        yolov8_engine_path,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold
    )
    
    print("加载姿态估计引擎...")
    pose_estimator = PoseEstimator(
        pose_engine_path,
        confidence_threshold=0.3
    )
    
    # 创建输入输出队列
    detection_input_queue = Queue(maxsize=10)
    detection_output_queue = Queue(maxsize=10)
    pose_input_queue = Queue(maxsize=10)
    pose_output_queue = Queue(maxsize=10)
    
    # 创建停止事件
    stop_event = threading.Event()
    
    # 创建推理线程
    print("启动推理线程...")
    detection_thread = threading.Thread(
        target=model_inference_thread,
        args=(detector, detection_input_queue, detection_output_queue, stop_event),
        daemon=True
    )
    
    pose_thread = threading.Thread(
        target=model_inference_thread,
        args=(pose_estimator, pose_input_queue, pose_output_queue, stop_event),
        daemon=True
    )
    
    detection_thread.start()
    pose_thread.start()
    
    # 打开摄像头
    print("打开摄像头...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        stop_event.set()
        return
    
    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # 用于计算FPS
    fps_counter = 0
    fps_start_time = time.time()
    fps = 0
    
    # 存储结果的字典
    detection_results = []
    pose_results = {}
    
    print("按 'q' 键退出...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("无法获取帧")
                break
            
            # 计算FPS
            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:
                fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
            
            # 发送帧到目标检测队列
            if not detection_input_queue.full():
                detection_input_queue.put({
                    "frame": frame.copy(),
                    "timestamp": time.time()
                })
            
            # 获取目标检测结果
            while not detection_output_queue.empty():
                detection_result = detection_output_queue.get()
                detection_results = detection_result["results"]
                detection_inference_time = detection_result["inference_time"]
                detection_output_queue.task_done()
                
                # 如果有人员检测结果，发送到姿态估计队列
                if detection_results and not pose_input_queue.full():
                    # 只处理置信度最高的人员检测结果
                    person_detections = [d for d in detection_results if d["class_name"] == "person"]
                    if person_detections:
                        # 选择置信度最高的人员
                        best_person = max(person_detections, key=lambda x: x["confidence"])
                        pose_input_queue.put({
                            "frame": frame.copy(),
                            "detection": best_person,
                            "timestamp": time.time()
                        })
            
            # 获取姿态估计结果
            while not pose_output_queue.empty():
                pose_result = pose_output_queue.get()
                pose_results[pose_result["timestamp"]] = pose_result
                pose_output_queue.task_done()
            
            # 绘制目标检测结果
            if detection_results:
                for result in detection_results:
                    x1, y1, x2, y2 = result["bbox"]
                    class_name = result["class_name"]
                    confidence = result["confidence"]
                    
                    # 绘制边界框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # 绘制类别名称和置信度
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # 绘制姿态估计结果
            if pose_results:
                # 获取最新的姿态估计结果
                latest_timestamp = max(pose_results.keys())
                latest_pose = pose_results[latest_timestamp]
                
                if latest_pose["results"]:
                    frame = pose_estimator.draw_poses(frame, latest_pose["results"])
            
            # 显示性能信息
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            if 'detection_inference_time' in locals():
                cv2.putText(frame, f"Detection Time: {detection_inference_time:.2f} ms", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            if pose_results and latest_pose:
                cv2.putText(frame, f"Pose Time: {latest_pose['inference_time']:.2f} ms", (10, 110), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            
            # 显示结果
            cv2.imshow("Multi-model Concurrent Inference", frame)
            
            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        # 清理资源
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()
        
        # 等待线程结束
        detection_thread.join()
        pose_thread.join()
        
        print("程序已退出")

if __name__ == "__main__":
    main()
