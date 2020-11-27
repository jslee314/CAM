import tensorflow as tf
from PIL import Image
import numpy as np
import cv2



class GradCAM:
    def __init__(self, instance, sample_size):
        self.instance = instance
        self.sample_size = sample_size
        self.image_size = (224, 224)  # (width, height)
        self.num_classes = 2  # class 개수

    def normalize(self, x):
        return tf.div(x, tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.sqrt(
            tf.reduce_mean(tf.square(x), axis=(1, 2, 3))) + tf.constant(1e-5), axis=-1), axis=-1), axis=-1))

    def build(self):

        with tf.variable_scope('grad_cam'):

            cam_layer = self.instance.cam_layer
            loss = tf.reduce_mean(tf.multiply(self.instance.logits, self.instance.prob), axis=1)
            grads = tf.gradients(ys=loss, xs=cam_layer)[0]  # (B, H, W, C)
            norm_grads = self.normalize(grads)

            weights = tf.reduce_mean(input_tensor=norm_grads, axis=(1, 2))
            weights = tf.expand_dims(tf.expand_dims(weights, axis=1), axis=1)
            height, width = cam_layer.get_shape().as_list()[1: 3]
            cam = tf.ones(shape=[self.sample_size, height, width], dtype=tf.float32)
            cam = tf.add(cam, tf.reduce_sum(input_tensor=tf.multiply(weights, cam_layer), axis=-1))
            self.cam = tf.maximum(cam, 0, name='outputs')

    def visualize(self, x, file_names):
        cam_output = self.instance.sess.run(self.cam,
                                            feed_dict={self.instance.x: x,
                                                       self.instance.training: False})
        cam_list = []

        for idx in range(self.sample_size):
            cam_output[idx] = cam_output[idx] / np.max(cam_output[idx])
            cam_list.append(cv2.resize(cam_output[idx], self.image_size))

        outputs = []

        for idx in range(self.sample_size):
            img = Image.open(file_names[idx], mode='r').convert('RGB')
            img = cv2.resize(np.asarray(img), self.image_size, interpolation=cv2.INTER_NEAREST)
            img = img.astype(float)
            img /= 255.

            img_cam = 255 * cam_list[idx]
            img_cam = np.uint8(img_cam)
            img_cam = cv2.applyColorMap(img_cam, cv2.COLORMAP_JET)
            img_cam = cv2.cvtColor(img_cam, cv2.COLOR_BGR2RGB)


            '''Grad-CAM 과 원본 이미지 중첩'''
            alpha = 0.0025
            img_cam = alpha * img_cam
            output = img + img_cam
            output /= output.max()

            #outputs.append(output)
            outputs.append(output)

            #jimage = Image.fromarray(img_cam)
            #jimage.save('D:\\Data\\cam_test\\img_cam.png')




        return outputs