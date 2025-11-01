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
from enum import Enum, auto
import base64

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
                data["image_uint8_array"] = ((request.image_uint8_array))
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
    boxes_np = np.asarray(boxes, dtype=float)
    logits_np = np.asarray(logits, dtype=float)

    scale = np.array([w, h, w, h], dtype=float)
    boxes_px = boxes_np * scale  # cx, cy, width, height in pixels

    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)

    colors = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
    ]

    for idx, (box, logit, phrase) in enumerate(zip(boxes_px, logits_np, phrases)):
        cx, cy, bw, bh = box
        x1 = int(max(cx - bw / 2, 0))
        y1 = int(max(cy - bh / 2, 0))
        x2 = int(min(cx + bw / 2, w - 1))
        y2 = int(min(cy + bh / 2, h - 1))

        color = colors[idx % len(colors)]
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

        label = f"{phrase} {logit:.2f}"
        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_w, text_h = text_size
        text_bg_tl = (x1, max(y1 - text_h - 4, 0))
        text_bg_br = (x1 + text_w + 4, max(y1, text_h + 4))
        cv2.rectangle(annotated_frame, text_bg_tl, text_bg_br, color, thickness=-1)
        cv2.putText(
            annotated_frame,
            label,
            (x1 + 2, text_bg_br[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        center = (int(round(cx)), int(round(cy)))
        bottom_center = (int(round(cx)), int(round(cy + bh / 2)))
        cv2.circle(annotated_frame, center, 5, (0, 0, 255), thickness=-1)
        cv2.circle(annotated_frame, bottom_center, 5, (0, 0, 255), thickness=-1)

    return annotated_frame

def dino_client_detect(model, image, prompt, box_threshold=0.55, text_threshold=0.5) -> DetObject:
    start_time = time.time()
    if not hasattr(dino_client_detect, "annotate_index"):
        dino_client_detect.annotate_index = 0
    request = DinoClientRequest(
        prompt=prompt,
        image_uint8_array=image.flatten().tolist(),
        image_shape=image.shape,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    model.send_request(request)
    request.wait_until_done(timeout=10) 
    end_time = time.time()
    print("receive result time: ", end_time)
    print("dino_client_detect 1 cost: ", end_time - start_time)
    boxes, logits, phrases = request.boxes, request.logits, request.phrases
    #annotated_frame = annotate(image_source=image, boxes=boxes, logits=logits, phrases=phrases)
    
    # 使用并递增函数属性
    # filename = f"images/frame_{dino_client_detect.annotate_index}_annotated_{prompt}.png"
    #cv2.imwrite(filename, annotated_frame)
    dino_client_detect.annotate_index += 1

    if boxes.shape[0] == 0:
        return []
    
    # 创建DetObject列表
    det_objects = []
    for i in range(len(boxes)):
        if prompt != "Pink bowl" and boxes[i][2] > 0.2 and boxes[i][3] > 0.2 : # 过滤掉高度过大的框
            continue
        det_obj = DetObject(boxes[i], logits[i], phrases[i])
        det_objects.append(det_obj)
    
    # 按 logits 从大到小排序
    det_objects.sort(key=lambda x: x.logit, reverse=True)
    print("dino debug:", boxes, logits, phrases)
    end_time = time.time()
    print("dino_client_detect 2 cost: ", end_time - start_time)
    
    return det_objects


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
