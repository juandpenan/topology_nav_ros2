from PIL import Image as Pimage

from cv_bridge import CvBridge

import rclpy

from rclpy.node import Node

from sensor_msgs.msg import Image

import torch.nn.functional as F

from vqa_msgs.msg import VisualFeatures

from vqa_ros import utils


class VQAModel(Node):

    def __init__(self):
        super().__init__('vqa_node')
        # params
        self.declare_parameter('frequency_execution_time', 2.1)
        self.declare_parameter('questions', ['where am i?'])

        self.questions = self.get_parameter(
            'questions').get_parameter_value().string_array_value

        self.model = utils.model
        self.image_data = None
        self.image_converter = CvBridge()

        # publishers
        self.feature_publisher = self.create_publisher(VisualFeatures, '/vqa/features', 1)
        # subscribers
        self.scene_data_subscriber = self.create_subscription(
            Image,
            '/image',
            self.scene_callback,
            1)

    def execute_model(self):
        try:
            image = self.image_converter.imgmsg_to_cv2(self.image_data)
            image = Pimage.fromarray(image)
            if self.feature_publisher.get_subscription_count() == 0:
                return
        except Exception:
            self.get_logger().warning('Waiting for image callback')
        else:

            visual_features = VisualFeatures()
            answers = []
            confidence = []

            for question in self.questions:
                if question.find('*') != -1:
                    question = question.replace('*', answers[1])
                    encoding = utils.processor(image, question, return_tensors="pt")
                    encoding = {k: v.to('cuda') for k, v in encoding.items()}
                    # forward pass
                    outputs = self.model(**encoding)
                    logits = outputs.logits
                    # Get the predicted label and its corresponding logit score
                    predicted_idx = logits.argmax(-1).item()                    
                    answer = self.model.config.id2label[predicted_idx]
                    conf = F.softmax(logits, dim=-1)[0][predicted_idx].item()
                else:
                                
                    encoding = utils.processor(image, question, return_tensors="pt")
                    encoding = {k: v.to('cuda') for k, v in encoding.items()}
                    outputs = self.model(**encoding)
                    logits = outputs.logits
                    # Get the predicted label and its corresponding logit score
                    predicted_idx = logits.argmax(-1).item()                   
               
                    answer = self.model.config.id2label[predicted_idx]
                    conf = F.softmax(logits, dim=-1)[0][predicted_idx].item()

                self.get_logger().debug('current question' + question)

                answers.append(answer)
                confidence.append(conf)

            visual_features.header.stamp = self.image_data.header.stamp
            visual_features.acc = confidence
            visual_features.data = answers
            visual_features.info.question_qty = len(self.questions)
            self.feature_publisher.publish(visual_features)    

    def scene_callback(self, data):
        self.image_data = data
        self.execute_model()


def main(args=None):
    rclpy.init(args=args)
    vqa_node = VQAModel()
    rclpy.spin(vqa_node)


if __name__ == '__main__':
    main()
