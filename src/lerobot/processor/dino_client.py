#!/usr/bin/env python
import asyncio
import websockets
import json
import threading
from collections import deque
import pickle
import time
import numpy as np
import cv2
from typing import Tuple, List
import supervision as sv

from enum import Enum, auto

class DinoRequestResult(Enum):
    INIT = 0
    PROCESSING = 1
    COMPLETED = 2
    FAILED = 3


global request_id_counter
request_id_counter = 0

class CameraIntrinsics:
    """相机内参数据结构"""
    fx: float  # x方向焦距
    fy: float  # y方向焦距
    cx: float  # 主点x坐标
    cy: float  # 主点y坐标

class DinoClientRequest:
    def __init__(self, prompt, image_uint8_array, image_shape, box_threshold, text_threshold, 
                 image_path=None):
        # Dino Server could accept either image as uint8 array or image path, you can pick one of them.
        # if we provide image_uint8_array, image_path should be None, vice versa.
        # when image_uint8_array is provided, image_shape must be provided as well, DINO will process from buffer
        # otherwise, it will load image from image_path
        global request_id_counter
        request_id_counter += 1
        self.id = request_id_counter
        self.prompt = prompt
        self.image_uint8_array = image_uint8_array
        self.image_shape = image_shape
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.image_path = image_path
        self.event  = threading.Event()
        self.state = DinoRequestResult.INIT

        self.boxes = None
        self.logits = None
        self.phrases = None

    def wait_until_done(self, timeout=None):
        if self.event is not None:
            return self.event.wait(timeout)
        return False

    def is_request_done(self):
        return self.state in (DinoRequestResult.COMPLETED, DinoRequestResult.FAILED)
    
    def get_request_state(self):
        return self.state

class DinoClientManager:
    def __init__(self, queue_size=1000, server_uri="ws://localhost:9548"):
        self.request_queue = deque(maxlen=queue_size)
        self.queue_lock = threading.Lock()
        self.event = threading.Event()
        self.result_lock = threading.Lock()
        self.latest_result = None
        self.request_id_counter = 0
        self.results = {}
        self.results_lock = threading.Lock()
        self.server_uri = server_uri
        self.worker = threading.Thread(target=self.client_worker, daemon=True)
        self.worker.start()

    def next_id(self):
        self.request_id_counter += 1
        return self.request_id_counter

    def send_request(self, request):
        self.queue_lock.acquire()
        self.request_queue.append(request)
        self.queue_lock.release()
        self.event.set()  # signal worker
        return request

    def client_worker(self):
        while True:
            self.event.wait()   # Wait until producer signals
            self.event.clear()  # Reset event

            # Pump all requests from queue
            while True:
                self.queue_lock.acquire()
                if len(self.request_queue) == 0:
                    self.queue_lock.release()
                    break

                request = self.request_queue.popleft()
                self.queue_lock.release()

                # Process request
                asyncio.run(self.remote_dino_job_start(request))

    async def remote_dino_job_start(self, request):
        async with websockets.connect(self.server_uri) as websocket:
            data = {"id": request.id, "prompt": request.prompt}
            if request.image_uint8_array is not None:
                data["image_uint8_array"] = request.image_uint8_array
                data["image_shape"] = request.image_shape
            elif request.image_path is not None:
                data["image_path"] = request.image_path
            data["box_threshold"] = request.box_threshold
            data["text_threshold"] = request.text_threshold
            print(f"client send start: {time.time()}")
            await websocket.send(json.dumps(data))
            print(f"client send end: {time.time()}")
            response = await websocket.recv()
            id, boxes, logits, phrases = pickle.loads(response)
            print(f"[Client] Server returned for id: {request.id}, time: {time.time()}")
            request.state = DinoRequestResult.COMPLETED
            request.boxes = boxes
            request.logits = logits
            request.phrases = phrases
            if request.event is not None:
                request.event.set()  # signal completion

    @staticmethod
    def get_image_as_uint8_array(image_path):
        import cv2
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from path: {image_path}")
        return image.flatten().tolist()


class DetObject:
    def __init__(self, box, logit, phrase):
        self.box = box
        self.logit = logit
        self.phrase = phrase


def annotate(image_source: np.ndarray, boxes: np.ndarray, logits: np.ndarray, phrases: List[str]) -> np.ndarray:
    """    
    This function annotates an image with bounding boxes and labels.

    Parameters:
    image_source (np.ndarray): The source image to be annotated.
    boxes (torch.Tensor): A tensor containing bounding box coordinates.
    logits (torch.Tensor): A tensor containing confidence scores for each bounding box.
    phrases (List[str]): A list of labels for each bounding box.

    Returns:
    np.ndarray: The annotated image.
    """
    h, w, _ = image_source.shape
    boxes = boxes * np.ndarray([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    detections = sv.Detections(xyxy=xyxy)

    labels = [
        f"{phrase} {logit:.2f}"
        for phrase, logit
        in zip(phrases, logits)
    ]

    bbox_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = bbox_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    # 定义圆的参数
    radius = 5           # 半径
    color = (0, 0, 255)  # 颜色（B, G, R），这里是红色
    thickness = -1       # 线宽，-1表示填充整个圆

    # 绘制实心圆
    for i in range(len(boxes)):
        cv2.circle(annotated_frame, (int(boxes[i, 0]), int(boxes[i, 1])), radius, color, thickness)
        cv2.circle(annotated_frame, (int(boxes[i, 0]), int(int(boxes[i, 1]) + int(boxes[i, 3]) / 2)), radius, color, thickness)

    return annotated_frame

def get_depth(x, y, depth_image, radius=3):
    """
    获取目标坐标附近某个范围的平均深度
    
    Args:
        x: 目标x坐标
        y: 目标y坐标
        depth_image: 深度图像
        radius: 搜索半径，默认为5像素
        
    Returns:
        平均深度值，如果没有有效深度则返回0
    """
    height, width, _ = depth_image.shape
    
    # 确保坐标在图像范围内
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    
    # 定义搜索范围
    x_min = max(0, x - radius)
    x_max = min(width, x + radius + 1)
    y_min = max(0, y - radius)
    y_max = min(height, y + radius + 1)
    
    # 提取区域内的深度值
    depth_region = depth_image[y_min:y_max, x_min:x_max]
    
    # 过滤深度为0的异常值
    valid_depths = depth_region[depth_region > 0]
    
    if len(valid_depths) == 0:
        return 0
    
    # 返回平均深度
    return np.mean(valid_depths)

def image_to_camera(x, y, 
                    depth: float,
                    intrinsics) -> np.ndarray:
    """
    图像坐标到相机坐标转换
    
    Args:
        image_points: 图像坐标点 (N, 2) 或 (2,)
        depth: 深度值 (米)
        undistort: 是否去畸变
        
    Returns:
        相机坐标点 (N, 3) 或 (3,)
    """

    # 像素坐标到归一化坐标
    x_norm = (x - intrinsics.cx) / intrinsics.fx
    y_norm = (y - intrinsics.cy) / intrinsics.fy

    # 乘以深度得到相机坐标
    x_cam = x_norm * depth
    y_cam = y_norm * depth
    z_cam = depth
    return np.array([x_cam, y_cam, z_cam])


def dino_client_detect(model, image, prompt, box_threshold=0.55, text_threshold=0.5) -> DetObject:
    start_time = time.time()
    if not hasattr(dino_client_detect, "annotate_index"):
        dino_client_detect.annotate_index = 0
    request = DinoClientRequest(
        prompt=prompt,
        image_uint8_array=None,
        image_shape=None,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    model.send_request(request)
    request.wait_until_done(timeout=10) 
    end_time = time.time()
    print("receive result time: ", end_time)
    print("dino_client_detect 1 cost: ", end_time - start_time)
    boxes, logits, phrases = request.boxes, request.logits, request.phrases
    annotated_frame = annotate(image_source=image, boxes=boxes, logits=logits, phrases=phrases)
    
    # 使用并递增函数属性
    filename = f"images/frame_{dino_client_detect.annotate_index}_annotated_{prompt}.png"
    cv2.imwrite(filename, annotated_frame)
    dino_client_detect.annotate_index += 1

    if boxes.shape[0] == 0:
        return []
    
    # 创建DetObject列表
    det_objects = []
    for i in range(len(boxes)):
        det_obj = DetObject(boxes[i], logits[i], phrases[i])
        det_objects.append(det_obj)
    
    # 按 logits 从大到小排序
    det_objects.sort(key=lambda x: x.logit, reverse=True)
    print("dino debug:", boxes, logits, phrases)
    end_time = time.time()
    print("dino_client_detect 2 cost: ", end_time - start_time)
    
    return det_objects

def dino_detect_front(model, image, prompt, depth, camera_param, assume_box = None, threshold=0.4):
    objects = dino_client_detect(model=model, image=image, prompt=prompt, box_threshold=threshold, text_threshold=threshold - 0.05)
    
    if len(objects) == 0:
        print("No objects detected")
        return None

    index = -1
    min_diff = float('inf')
    if (assume_box != None):
        for i in range(len(objects)):
            if (objects[i].box[2] / assume_box[2] < 0.7 or objects[i].box[2] / assume_box[2] > 1.6):
                continue
            if (abs(objects[i].box[0] - assume_box[0]) > 0.2):
                continue
            if (abs(objects[i].box[0] - assume_box[0]) < min_diff):
                min_diff = abs(objects[i].box[0] - assume_box[0])
                index = i

    else:
        index = 0
    
    if index == -1:
        print("No matching object found")
        return None

    # 获取检测框中心点
    x_center = int(objects[index].box[0] * 640)
    y_center = int(objects[index].box[1] * 480) + int(objects[index].box[3] * 480 / 2)
    # 获取深度
    z = get_depth(x_center, y_center, depth)
    # 转换到相机坐标系
    cam_p = image_to_camera(x_center, y_center, z, camera_param.camera_intrinsics)

    return cam_p



if __name__ == "__main__":
    client = DinoClientManager()

    # Example: main thread pushes requests
    import time
    picture_prompts = [
        ("/home/robot/zhx/script/tmp/left.png", (640,480,3), "blue square",0.45,0.25),
        ("/home/robot/zhx/script/tmp/right.png", (640,480,3), "blue square",0.45,0.25),
    ]
    request_left = None
    request_right = None
    for picture, shape, prompt, box_threshold, text_threshold in picture_prompts:
        image_uint8_array = client.get_image_as_uint8_array(picture)
        request = DinoClientRequest(
            prompt=prompt,
            image_uint8_array=image_uint8_array,
            image_shape=shape,
            box_threshold=box_threshold,
            text_threshold=text_threshold
        )

        client.send_request(request)
        print(f"Request ID: {request.id} for picture {picture}")

        if not request_left:
            request_left = request
        else:
            request_right = request

    if request_left:
        while not request_left.is_request_done():
            print("waiting for left request to complete...")
            time.sleep(0.1)

    if request_right:
        request_right.wait_until_done(timeout=10)

    print(f"Left Result: ID={request_left.id}, Boxes={request_left.boxes}, Logits={request_left.logits}, Phrases={request_left.phrases}")
    print(f"Right Result: ID={request_right.id}, Boxes={request_right.boxes}, Logits={request_right.logits}, Phrases={request_right.phrases}")
