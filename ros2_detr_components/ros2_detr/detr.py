from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# you can specify the revision tag if you don't want the timm dependency
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.9
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
    )

from rclpy.node import Node

class DETR(Node):

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
                GraspPose, 
                'get_gr_convnet_grasp_pose', 
                self.grasp_pose_callback,
                )
        
        # track latest image
        self._latest_rgb_image = None

        self.logger.info("DETR object detection service ready.")

    def _image_callback(self, rgb):
        """Callback function for image topic"""
        self._latest_rgb_image = rgb

    def grasp_pose_callback(self, request, response):
        self.get_logger().info('Computing grasp pose...')

        # preprocess the latest RGB, depth images for the GRCNN
        bgra_img = self.cv_bridge.imgmsg_to_cv2(self._latest_rgb_image, "bgra8")
        rgb_img = cv2.cvtColor(bgra_img, cv2.COLOR_BGRA2RGB)


        # crop (360, 640) image about center 
        rgb_img = rgb_img[60:300, 120:540]

        # resize RGB image to 244x244
        rgb_img = cv2.resize(rgb_img, (244, 244))
        
        # visualize RGB image
        cv2.imshow("rgb8", np.array(rgb_img, dtype=np.uint8))
        cv2.waitKey(1000)
        
        # normalize RGB image
        rgb_img = rgb_img.astype(np.float32) / 255.0
        rgb_img -= rgb_img.mean()

        depth_img = self.cv_bridge.imgmsg_to_cv2(self._latest_depth_image, "32FC1") # check encoding
        
        # check for nan and inf values in depth image
        print("NaNs: {}".format(np.isnan(depth_img).sum()))
        print("Infs: {}".format(np.isinf(depth_img).sum()))

        cv2.imshow("depth", depth_img)
        cv2.waitKey(1000)
       
        # inpaint nan and inf values in depth image
        nan_mask = np.isnan(depth_img)
        inf_mask = np.isinf(depth_img)
        mask = np.logical_or(nan_mask, inf_mask)
        mask = cv2.UMat(mask.astype(np.uint8))

        # Scale to keep as float, but has to be in bounds -1:1 to keep opencv happy.
        scale = np.ma.masked_invalid(np.abs(depth_img)).max()
        depth_img = depth_img.astype(np.float32) / scale  # Has to be float32, 64 not supported.
        depth_img = cv2.inpaint(depth_img, mask, 1, cv2.INPAINT_NS)
        # interpolate remaining nan values with nearest neighbor
        depth_img = np.array(depth_img.get())
        y, x = np.where(~np.isnan(depth_img))
        x_range, y_range = np.meshgrid(np.arange(depth_img.shape[1]), np.arange(depth_img.shape[0]))
        depth_img = griddata((x, y), depth_img[y, x], (x_range, y_range), method='nearest')
        depth_img = depth_img * scale

        cv2.imshow("depth_inpaint", depth_img)
        cv2.waitKey(1000)

        # resize depth image to 244x244
        depth_img = depth_img[60:300, 120:540]
        depth_img = cv2.resize(depth_img, (244, 244))

        # normalize depth image
        depth_img = np.clip((depth_img - depth_img.mean()), -1, 1) 
        depth_img = np.expand_dims(depth_img, axis=-1)

        self.logger.info("RGB image:")
        self.logger.info("{}".format(rgb_img.shape))
        self.logger.info("Depth image:")
        self.logger.info("{}".format(depth_img.shape))

        # combine RGB and depth images into numpy array without using preprocess
        img = np.concatenate((depth_img, rgb_img), axis=2)
        img = np.expand_dims(img, axis=0)
        img = np.transpose(img, (0, 3, 1, 2))
        img = img.astype(np.float32)

        self.logger.info("GRCNN input:")
        self.logger.info("{}".format(img.shape))
        
        # run the image through the GRCNN using onnxruntime
        outputs = self.grcnn.run(None, {"input": img}) # TODO: check input format
        self.logger.info("GRCNN output:")
        self.logger.info("{}".format(outputs[0].shape))
        self.logger.info("{}".format(outputs[1].shape))
        self.logger.info("{}".format(outputs[2].shape))
        self.logger.info("{}".format(outputs[3].shape))
        self.logger.info("{}".format(len(outputs)))

        # post process the output
        q_img, ang_img, width_img = post_process_output(*outputs)
        
        cv2.imshow("q_img", cv2.applyColorMap(cv2.normalize(q_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET))
        cv2.waitKey(1000)
        
        cv2.imshow("ang_img", cv2.applyColorMap(cv2.normalize(ang_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET))
        cv2.waitKey(1000)

        cv2.imshow("width_img", cv2.applyColorMap(cv2.normalize(width_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLORMAP_JET))
        cv2.waitKey(1000)

        outputs = [q_img, ang_img, width_img]
        #for output in outputs:
        #    heatmap = output[0]
        #    heatmap = np.transpose(heatmap, (1, 2, 0))
        #    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        #    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        #    cv2.imshow("q_img", heatmap)
        #    cv2.waitKey(1000)

        grasps = detect_grasps(q_img, ang_img, width_img)
        
        self.logger.info("Grasps:")
        self.logger.info("{}".format(grasps))
        
        print(grasps[0].as_gr.angle)

        # convert grasp data to world coordinates
        grasp_pose = PoseStamped()
        response.grasps = [grasp_pose]
        
        return response

def main(args=None):
    rclpy.init(args=args)
    grasp_pose_service = GRCNN()
    rclpy.spin(grasp_pose_service)
    grasp_pose_service.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()