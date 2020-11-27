from DeepLearning.CAM.model import *
from DeepLearning.CAM.constant import *
from DeepLearning.CAM.grad_cam import GradCAM

import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


class Neuralnet:

    def cam_test(self):
        sample_num = 5
        class_num = 2
        batch_size = sample_num * class_num
        img_size = (224, 224)
        sample_path = 'D:\\Data\\gelontoxon_cam'

        def save_matplot_img(outputs, sample_num, class_num, file_name):
            f = plt.figure(figsize=(10, 8))
            plt.suptitle('CAM (Class Activation Mapping)', fontsize=20)
            outer = gridspec.GridSpec(1, 1, wspace=0.2, hspace=0.2)

            inner = gridspec.GridSpecFromSubplotSpec(class_num, sample_num, subplot_spec=outer[0], wspace=0.1, hspace=0.1)

            for cls in range(class_num):
                for sample in range(sample_num):
                    subplot = plt.Subplot(f, inner[sample + cls * sample_num])
                    subplot.axis('off')
                    subplot.imshow(outputs[sample + cls * sample_num])
                    f.add_subplot(subplot)

            f.savefig(os.path.join('D:\\Data\\cam_test', file_name))
            print('>> CAM Complete')

        def get_file_names():
            file_names, labels = [], []

            for cls in range(class_num):
                file_name = [[os.path.join(path, file) for file in files] for path, dir, files in os.walk(os.path.join(sample_path, str(cls)))]
                file_name = np.asarray(file_name).flatten()

                random_sort = np.random.permutation(file_name.shape[0])
                file_name = file_name[random_sort][:sample_num]

                for f_name in file_name:
                    file_names.append(f_name)
                    labels.append(cls)

            file_names = np.asarray(file_names)
            labels = np.asarray(labels)

            file_names = tf.convert_to_tensor(file_names, dtype=tf.string)
            labels = tf.convert_to_tensor(labels, dtype=tf.int32)

            return file_names, labels

        def normal_data(x, y):
            with tf.variable_scope('normal_data'):
                data = tf.read_file(x)
                data = tf.image.decode_png(data, channels=3, name='decode_img')
                data = tf.image.resize_images(data, size=img_size)
                data = tf.divide(data, 255.)
            return data, y

        def data_loader(file_names, labels):
            with tf.variable_scope('data_loader'):
                dataset = tf.contrib.data.Dataset.from_tensor_slices((file_names, labels)).repeat()

                normal_dataset_map = dataset.map(normal_data).batch(batch_size)
                normal_iterator = normal_dataset_map.make_one_shot_iterator()
                normal_batch_input = normal_iterator.get_next()

            return normal_batch_input

        '''>> Start'''
        file_names, labels = get_file_names()
        batch_data = data_loader(file_names, labels)

        with tf.Session() as sess:
            file_names_output = sess.run(file_names)
            batch_xy = sess.run(batch_data)

            model = Model(sess=sess, name='model')
            model.build()

            grad_cam = GradCAM(instance=model, sample_size=sample_num * class_num)
            grad_cam.build()

            '''>> 학습된 파라미터 로드'''
            self._saver = tf.train.Saver()
            ckpt_st = tf.train.get_checkpoint_state(os.path.join(flags.FLAGS.trained_param_path))

            if ckpt_st is not None:
                self._saver.restore(sess, ckpt_st.model_checkpoint_path)
                print('>> Model Restored')

            grad_outputs = grad_cam.visualize(batch_xy[0], file_names_output)

            save_matplot_img(grad_outputs, sample_num, class_num, 'grad_cam.png')


neuralnet = Neuralnet()
neuralnet.cam_test()















