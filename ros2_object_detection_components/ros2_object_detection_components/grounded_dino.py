"""
A node for performing object detection with grounded dino.
"""

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch
from PIL import Image
import requests

from rclpy.node import Node
from ros2_object_detection_msgs.msg import BoundingBoxes, BoundingBox
from ros2_object_detection_msgs.srv import DetectObjectGroundedDino


class GroundedDino(Node):

    def __init__(self, image_topic: str):
        super().__init__('grounded_dino_object_detection')
        self.logger = self.get_logger() # instantiate logger

        # pull image processor and detr model from huggingface
        model_id = "IDEA-Research/grounding-dino-tiny"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

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
            callback_group=self.camera_callback_group,
            )

        # create service for requesting object detection
        self.srv = self.create_service(
                DetectObject, 
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
        confidence_threshold = request.confidence

        # convert ROS image message to opencv
        bgra_img = self.cv_bridge.imgmsg_to_cv2(self._latest_rgb_image, "rgb8")
        rgb_img = cv2.cvtColor(bgra_img, cv2.)
        
        # perform Grounded Dino inference
        self.get_logger().info('Running Grounded Dino Inference...')
        inputs = self.processor(images=self._latest_rgb_image, text=text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # parse predictions
        results = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=0.4, # TODO: read this from config
                text_threshold=0.3, # TODO: read this from config
                target_sizes=[self._latest_rgb_image.size[::-1]]
            )

        #for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                # TODO: complete the response message
                # box = [round(i, 2) for i in box.tolist()]
                # print(
                #         f"Detected {model.config.id2label[label.item()]} with confidence "
                #         f"{round(score.item(), 3)} at location {box}"
                # )
        
        # return parsed predictions data
        
        return response

def main(args=None):
    rclpy.init(args=args)
    grounded_dino = GroundedDino(image_topic='placeholder') # move to args when finished debugging
    rclpy.spin(grounded_dino)
    detr.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()