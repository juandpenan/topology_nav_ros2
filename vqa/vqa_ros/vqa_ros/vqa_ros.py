from PIL import Image as Pimage

from cv_bridge import CvBridge

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image

import torch

from vqa_msgs.msg import VisualFeatures

from vqa_ros import utils

torch.set_grad_enabled(False)


class VQAModel(Node):

    def __init__(self):
        super().__init__('vqa_node')
        # params
        self.declare_parameter('frequency_execution_time', 2.1)
        self.declare_parameter('questions', ['where am i?'])
        execution_time = self.get_parameter(
            'frequency_execution_time').get_parameter_value().double_value
        self.questions = self.get_parameter(
            'questions').get_parameter_value().string_array_value
        self.model = None
        self.load_model()
        self.image_data = None
        self.image_converter = CvBridge()

        # every execution_time seconds
        self.timer = self.create_timer(execution_time, self.timer_callback)
        # publishers
        self.feature_publisher = self.create_publisher(VisualFeatures, '/vqa/features', 3)
        # subscribers
        self.scene_data_subscriber = self.create_subscription(
            Image,
            '/image',
            self.scene_callback,
            1)

    def timer_callback(self):
        try:
            image = self.image_converter.imgmsg_to_cv2(self.image_data)
            if self.feature_publisher.get_subscription_count() == 0:
                return
        except Exception:
            self.get_logger().warning('Waiting for image callback')
        else:

            visual_features = VisualFeatures()
            answers = []
            confidence = []
            # TODO (juandpenan)
            # with Pool(5) as p:
            # answers,confidence = zip(*p.starmap(utils._plot_inference_qa, zip(repeat(image),self.questions)))
            for question in self.questions:
                if question.find('*') != -1:
                    question = question.replace('*', answers[1])
                    answer, conf = self._plot_inference_qa(image, question)
                else:
                    answer, conf = self._plot_inference_qa(image, question)

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

    def load_model(self):
        self.model = torch.hub.load('ashkamath/mdetr:main',
                                    'mdetr_efficientnetB5_gqa',
                                    pretrained=True,
                                    return_postprocessor=False)
        self.model = self.model.cuda()
        self.model.eval();
        return
    
    def _plot_inference_qa(self, im, caption):
        # mean-std normalize the input image (batch-size: 1)
        img_converted = Pimage.fromarray(im)
        img = utils._transform(img_converted).unsqueeze(0).cuda()

        # propagate through the model
        memory_cache = self.model(img, [caption], encode_and_save=True)
        outputs = self.model(img, [caption], encode_and_save=False, memory_cache=memory_cache)

        # Classify the question type
        type_conf, type_pred = outputs['pred_answer_type'].softmax(-1).max(-1)
        ans_type = type_pred.item()
        types = ['obj', 'attr', 'rel', 'global', 'cat']

        ans_conf, ans = outputs[f'pred_answer_{types[ans_type]}'][0].softmax(-1).max(-1)      
        answer = utils.id2answerbytype[f'answer_{types[ans_type]}'][ans.item()]

        conf = round(100 * type_conf.item() * ans_conf.item(), 2)

        return answer, conf

def main(args=None):
    rclpy.init(args=args)
    vqa_node = VQAModel()
    rclpy.spin(vqa_node)


if __name__ == '__main__':
    main()
