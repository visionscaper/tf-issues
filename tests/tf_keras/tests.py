import unittest

import tensorflow as tf

from tensorflow.keras import backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Masking, Dense, Activation, Dropout, TimeDistributed

from tensorflow.keras.layers import GRU
from tensorflow.compat.v1.keras.layers import CuDNNGRU

from tensorflow.keras.optimizers import Adam

from data.tf_keras.fake_chat_generator import FakeChatGenerator

from models.seq2seq_with_submodels import Seq2SeqWithSubmodels

# import pydevd_pycharm
# pydevd_pycharm.settrace('192.168.178.8', port=57491, stdoutToServer=True, stderrToServer=True)


class TestIssuesTFKeras(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.KERAS_CLASSES = {
            "Model": Model,

            "Input": Input,

            "Masking": Masking,
            "Dense": Dense,
            "Activation": Activation,
            "TimeDistributed": TimeDistributed,
            "Dropout": Dropout,

            "RNN": CuDNNGRU
        }

        tf.logging.set_verbosity(tf.logging.INFO)
        cls.TF_CONFIG = tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))

    def setUp(self):
        # E.g. character symbols
        self.num_unique_symbols = 96
        self.max_seq_length = 128
        self.batch_size = 64

        self.embedding_size = 16

        self.loss_type = 'categorical_crossentropy'

        K.clear_session()

        sess = tf.Session(config=TestIssuesTFKeras.TF_CONFIG)
        K.set_session(sess)

    def test_single_gpu_float32_no_masking_use_partially_known_dropout_noise_shape(self):
        dtype = self._enable_float32()

        batch_generator = FakeChatGenerator(
            num_unique_symbols=self.num_unique_symbols,
            max_seq_length=self.max_seq_length,
            batch_size=self.batch_size,
            return_sample_weights=True,
            dtype=dtype)

        model_generator = Seq2SeqWithSubmodels(TestIssuesTFKeras.KERAS_CLASSES,
                                               self.max_seq_length,
                                               self.num_unique_symbols,
                                               use_masking=False,
                                               use_partially_known_dropout_noise_shape=True,
                                               dtype=dtype)

        with tf.device('/gpu:0'):
            models = model_generator.stamp_train_model()

            model = models["train_model"]

        self._compile_model(model, lambda: Adam(lr=1e-4))

        # TODO : shapes are only correct when using one-hot encoding
        dataset = tf.data.Dataset.from_generator(
            batch_generator,
            ({
                 'chat-input': dtype,
                 'teacher-forcing-input': dtype
             },

             dtype,  # outputs
             dtype),  # sample weights
            ({
                 'chat-input': (None, None, None),
                 'teacher-forcing-input': (None, None, None)
             },
             (None, None, None),
             (None, None)))

        model.fit(dataset,
                  steps_per_epoch=len(batch_generator),
                  epochs=5,
                  verbose=1,
                  max_queue_size=10,
                  workers=3)

    def test_single_gpu_float32_no_masking_no_dropout_noise_shape(self):
        dtype = self._enable_float32()

        batch_generator = FakeChatGenerator(
            num_unique_symbols=self.num_unique_symbols,
            max_seq_length=self.max_seq_length,
            batch_size=self.batch_size,
            return_sample_weights=True,
            dtype=dtype)

        model_generator = Seq2SeqWithSubmodels(TestIssuesTFKeras.KERAS_CLASSES,
                                               self.max_seq_length,
                                               self.num_unique_symbols,
                                               use_masking=False,
                                               disable_dropout_noise_shape=True,
                                               dtype=dtype)

        with tf.device('/gpu:0'):
            models = model_generator.stamp_train_model()

            model = models["train_model"]

        self._compile_model(model, lambda: Adam(lr=1e-4))

        # TODO : shapes are only correct when using one-hot encoding
        dataset = tf.data.Dataset.from_generator(
            batch_generator,
            ({
                 'chat-input': dtype,
                 'teacher-forcing-input': dtype
             },

             dtype,  # outputs
             dtype),  # sample weights
            ({
                 'chat-input': (None, None, None),
                 'teacher-forcing-input': (None, None, None)
             },
             (None, None, None),
             (None, None)))

        model.fit(dataset,
                  steps_per_epoch=len(batch_generator),
                  epochs=5,
                  verbose=1,
                  max_queue_size=10,
                  workers=3)

    def test_single_gpu_float32_no_masking_calculate_dropout_noise_shape(self):
        dtype = self._enable_float32()

        batch_generator = FakeChatGenerator(
            num_unique_symbols=self.num_unique_symbols,
            max_seq_length=self.max_seq_length,
            batch_size=self.batch_size,
            return_sample_weights=True,
            dtype=dtype)

        model_generator = Seq2SeqWithSubmodels(TestIssuesTFKeras.KERAS_CLASSES,
                                               self.max_seq_length,
                                               self.num_unique_symbols,
                                               use_masking=False,
                                               use_partially_known_dropout_noise_shape=False,
                                               dtype=dtype)

        with tf.device('/gpu:0'):
            models = model_generator.stamp_train_model()

            model = models["train_model"]

        self._compile_model(model, lambda: Adam(lr=1e-4))

        # TODO : shapes are only correct when using one-hot encoding
        dataset = tf.data.Dataset.from_generator(
            batch_generator,
            ({
                 'chat-input': dtype,
                 'teacher-forcing-input': dtype
             },

             dtype,  # outputs
             dtype),  # sample weights
            ({
                 'chat-input': (None, None, None),
                 'teacher-forcing-input': (None, None, None)
             },
             (None, None, None),
             (None, None)))

        model.fit(dataset,
                  steps_per_epoch=len(batch_generator),
                  epochs=5,
                  verbose=1,
                  max_queue_size=10,
                  workers=3)

    def _compile_model(self, model, create_optimizer):

        optimizer = create_optimizer()

        model.compile(loss=self.loss_type,
                      sample_weight_mode="temporal",
                      optimizer=optimizer,
                      weighted_metrics=["accuracy"])

    def _enable_float32(self):
        dtype = 'float32'

        K.set_floatx(dtype)
        K.set_epsilon(1e-7)

        return dtype

    def _enable_float16(self):
        dtype = 'float16'

        K.set_floatx(dtype)
        K.set_epsilon(1e-4)

        return dtype


if __name__ == '__main__':
    unittest.main()
