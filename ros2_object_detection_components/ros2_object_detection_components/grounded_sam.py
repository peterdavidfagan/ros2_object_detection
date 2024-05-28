"""
A node for performing object detection with grounded dino.
"""

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
import numpy as np
import PIL as pil
from PIL import ImageDraw, ImageFont
import requests

import cv2
from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy

from sensor_msgs.msg import Image
from ros2_object_detection_msgs.msg import BoundingBox
from ros2_object_detection_msgs.srv import DetectObjectGroundedDino


class GroundedSam(Node):

    def __init__(self, image_topic: str):
        super().__init__('grounded_dino_object_detection')
        self.logger = self.get_logger() # instantiate logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # pull image processor and dino model from huggingface
        model_id = "IDEA-Research/grounding-dino-tiny"
        self.dino_processor = AutoProcessor.from_pretrained(model_id)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)     

        # pull image processor and sam model from huggingface
        self.sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
        self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

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
                DetectObjectGroundedDino, 
                'grounded_dino_detect_object', 
                self._detect_object,
                )
        
        # track latest image
        self._latest_rgb_image = None

        self.logger.info("Grounded Dino object detection service is ready.")

    def _image_callback(self, rgb):
        self._latest_rgb_image = rgb

    def _detect_object(self, request, response):
        # parse request
        text = request.prompt
        confidence_threshold = request.confidence_threshold

        # convert ROS image message to opencv
        rgb_img = np.array(self.cv_bridge.imgmsg_to_cv2(self._latest_rgb_image, "rgb8"))
        rgb_img = pil.Image.fromarray(rgb_img)

        # perform Grounded Dino inference
        self.get_logger().info('Running Grounded Sam Inference...')
        inputs = self.dino_processor(images=rgb_img, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.dino_model(**inputs)

        # parse predictions
        results = self.dino_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.4, # TODO: read this from config
                text_threshold=0.3, # TODO: read this from config
                target_sizes=[rgb_img.size[::-1]]
            )[0]

        # create bounding box response
        bboxs = []
        masks = []
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

            # generate mask with SAM
            center_point = [box[2]//2, box[3]//2]
            inputs = self.sam_processor(rgb_img, input_points=center_point, return_tensors="pt").to(self.device)
            with torch.no_grad():
                sam_outputs = self.sam_model(**inputs)

            masks = self.sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
            )

            # TODO: parse masks data


        # Draw bounding boxes on the image
        # TODO: draw segmentation on the image
        draw = ImageDraw.Draw(rgb_img)

        for bbox in bboxs:
            draw.rectangle([bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax], outline="red")
            # TODO: add label


        # Publish the annotated image
        annotated_img_msg = self.cv_bridge.cv2_to_imgmsg(np.array(rgb_img), encoding="rgb8")
        self.detection_publisher.publish(annotated_img_msg)        
        
        # return parsed predictions data
        response.bounding_boxes = bboxs
        response.masks = masks

        return response

def main(args=None):
    rclpy.init(args=args)
    grounded_sam = GroundedSam(image_topic='overhead_camera') # move to args when finished debugging
    rclpy.spin(grounded_sam)
    detr.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()