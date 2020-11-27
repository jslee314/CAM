import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('data_path', 'G:/04_dataset/gelontoxon', '데이터 경로')
flags.DEFINE_string('trained_param_path',
                    'D:\\PythonRepository\\source\\CAM\\train_log',
                    '훈련 체크포인트 로그 경로')


flags.DEFINE_integer('epochs', 500, '신경망 훈련 시 에폭 수')
flags.DEFINE_integer('batch_size', 10, '신경망 훈련 시 배치 크기')

flags.DEFINE_float('dropout_rate', 0.6, '신경망 드롭아웃 비율')
flags.DEFINE_float('learning_rate', 0.045, '신경망 학습 률')
flags.DEFINE_integer('decay_step', 6, '학습 률 decay step 크기')
flags.DEFINE_float('decay_rate', 0.98, '학습 률 decay rate')