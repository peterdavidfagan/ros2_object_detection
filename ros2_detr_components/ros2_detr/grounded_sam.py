"""
A node for performing object detection with grounded sam.
"""

from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests

from rclpy.node import Node

class GroundedSam(Node):

    def __init__(self):
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
            '/zed2i/zed_node/rgb/image_rect_color',
            self._image_callback,
            callback_group=self.camera_callback_group,
            )

        # create service for requesting object detection
        self.srv = self.create_service(
                DetectObject, 
                'detr_detect_object', 
                self._detect_object,
                )
        
        # track latest image
        self._latest_rgb_image = None

        self.logger.info("DETR object detection service is ready.")

    def _image_callback(self, rgb):
        self._latest_rgb_image = rgb

    def _detect_object(self, request, response):
        # convert ROS image message to opencv
        bgra_img = self.cv_bridge.imgmsg_to_cv2(self._latest_rgb_image, "rgb8")
        rgb_img = cv2.cvtColor(bgra_img, cv2.)
        
        # perform DETR inference
        self.get_logger().info('Running DETR Inference...')
        inputs = self.processor(images = rgb_img, return_tensors='pt') # preprocess image data
        predictions = self.model(**inputs)

        # parse predictions
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(predictions, target_sizes=target_sizes, threshold=0.9)[0]

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if label == "apple":
                box = [round(i, 2) for i in box.tolist()]
                print(
                        f"Detected {model.config.id2label[label.item()]} with confidence "
                        f"{round(score.item(), 3)} at location {box}"
                )
        
        # return parsed predictions data
        
        return response

def main(args=None):
    rclpy.init(args=args)
    detr = DETR()
    rclpy.spin(detr)
    detr.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()