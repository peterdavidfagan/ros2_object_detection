"""
A node for performing object detection with DETR.
"""

from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
import numpy as np
import PIL as pil
import requests

import cv2
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy

from sensor_msgs.msg import Image
from ros2_object_detection_msgs.msg import BoundingBox
from ros2_object_detection_msgs.srv import DetectObjectDETR 

class DETR(Node):

    def __init__(self, image_topic: str):
        super().__init__('detr_object_detection')
        self.logger = self.get_logger() # instantiate logger

        # pull image processor and detr model from huggingface
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

        # create a bridge between ROS2 and OpenCV
        self.cv_bridge = CvBridge()

        # ensure parallel execution of camera callbacks
        self.camera_callback_group = ReentrantCallbackGroup()

        # set QOS profile for camera image callback
        self.camera_qos_profile = QoSProfile(
                depth=1,
                history=QoSHistoryPolicy(rclpy.qos.HistoryPolicy.KEEP_LAST),
                reliability=QoSReliabilityPolicy(rclpy.qos.ReliabilityPolicy.RELIABLE),
            )

        # subscribe to camera image
        self.subscription = self.create_subscription(
            Image,
            image_topic,
            self._image_callback,
            self.camera_qos_profile,
            callback_group=self.camera_callback_group,
            )

        # publish latest object detections
        self.detection_publisher = self.create_publisher(Image, '/grounded_dino_detected_objects', 10)

        # create service for requesting object detection
        self.srv = self.create_service(
                DetectObjectDETR, 
                'detr_detect_object', 
                self._detect_object,
                )
        
        # track latest image
        self._latest_rgb_image = None

        self.logger.info("DETR object detection service is ready.")

    def _image_callback(self, rgb):
        self._latest_rgb_image = rgb

    def _detect_object(self, request, response):
        # parse request message data
        confidence_threshold = request.confidence_threshold
        class_name = request.object_class

        # convert ROS image message to opencv
        rgb_img = np.array(self.cv_bridge.imgmsg_to_cv2(self._latest_rgb_image, "rgb8"))
        rgb_img = pil.Image.fromarray(rgb_img)        

        # perform DETR inference
        self.get_logger().info('Running DETR Inference...')
        inputs = self.processor(images = rgb_img, return_tensors='pt') # preprocess image data
        predictions = self.model(**inputs)

        # parse predictions
        target_sizes = torch.tensor([rgb_img.size[::-1]])
        results = self.processor.post_process_object_detection(predictions, target_sizes=target_sizes, threshold=confidence_threshold)[0]

        bboxs = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            # complete bounding box message
            bbox = BoundingBox()
            bbox.confidence = float(score)
            bbox.object = label
            bbox.xmin = int(box[0])
            bbox.ymin = int(box[1])
            bbox.xmax = int(box[2])
            bbox.ymax = int(box[3])

            bboxs.append(bbox)
        
        # Draw bounding boxes on the image
        draw = ImageDraw.Draw(rgb_img)

        for bbox in bboxs:
            draw.rectangle([bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax], outline="red")
            # TODO: add label

        # Publish the annotated image
        annotated_img_msg = self.cv_bridge.cv2_to_imgmsg(np.array(rgb_img), encoding="rgb8")
        self.detection_publisher.publish(annotated_img_msg)        

        # return parsed predictions data
        response.bounding_boxes = bboxs

        return response

def main(args=None):
    rclpy.init(args=args)
    detr = DETR(image_topic='overhead_camera') # move image topic to args when finished debugging
    rclpy.spin(detr)
    detr.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()