import unittest

import logging

import tensorflow as tf

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

            # Should now be transparent, GRU still gives poorer performance
            "RNN": CuDNNGRU #GRU
        }

        logging.getLogger("tensorflow").setLevel(logging.INFO)

        cls.TF_CONFIG = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                                 gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))

    def setUp(self):
        # E.g. character symbols
        self.num_unique_symbols = 96
        self.max_seq_length = 128
        self.batch_size = 64

        self.embedding_size = 16

        self.loss_type = 'categorical_crossentropy'

        self.loss_scale = 128

        tf.compat.v1.keras.backend.clear_session()

        sess = tf.compat.v1.Session(config=TestIssuesTFKeras.TF_CONFIG)
        tf.compat.v1.keras.backend.set_session(sess)

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
                 'chat-input': (None, None, self.num_unique_symbols),
                 'teacher-forcing-input': (None, None, self.num_unique_symbols)
             },
             (None, None, self.num_unique_symbols),
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
                 'chat-input': (None, None, self.num_unique_symbols),
                 'teacher-forcing-input': (None, None, self.num_unique_symbols)
             },
             (None, None, self.num_unique_symbols),
             (None, None)))

        model.fit(dataset,
                  steps_per_epoch=len(batch_generator),
                  epochs=5,
                  verbose=1,
                  max_queue_size=10,
                  workers=3)

    def test_multi_gpu_float32_no_masking_no_dropout_noise_shape_no_sample_weight_mode(self):
        dtype = self._enable_float32()

        batch_generator = FakeChatGenerator(
            num_unique_symbols=self.num_unique_symbols,
            max_seq_length=self.max_seq_length,
            batch_size=self.batch_size,
            return_sample_weights=False,
            dtype=dtype)

        model_generator = Seq2SeqWithSubmodels(TestIssuesTFKeras.KERAS_CLASSES,
                                               self.max_seq_length,
                                               self.num_unique_symbols,
                                               use_masking=False,
                                               disable_dropout_noise_shape=True,
                                               dtype=dtype)

        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            models = model_generator.stamp_train_model()
            model = models["train_model"]

            self._compile_model(model, lambda: Adam(lr=1e-4), sample_weight_mode=False)

        # TODO : shapes are only correct when using one-hot encoding
        dataset = tf.data.Dataset.from_generator(
            batch_generator,
            ({
                 'chat-input': dtype,
                 'teacher-forcing-input': dtype
             },

             dtype),  # outputs
            ({
                 'chat-input': (None, None, self.num_unique_symbols),
                 'teacher-forcing-input': (None, None, self.num_unique_symbols)
             },
             (None, None, self.num_unique_symbols)))

        model.fit(dataset,
                  steps_per_epoch=len(batch_generator),
                  epochs=5,
                  verbose=1,
                  max_queue_size=10,
                  workers=3)

    def test_multi_gpu_float32_no_masking_no_dropout_noise_shape_sample_weight_mode(self):
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

        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            models = model_generator.stamp_train_model()
            model = models["train_model"]

            self._compile_model(model, lambda: Adam(lr=1e-4), sample_weight_mode=True)

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
                 'chat-input': (None, None, self.num_unique_symbols),
                 'teacher-forcing-input': (None, None, self.num_unique_symbols)
             },
             (None, None, self.num_unique_symbols),
             (None, None)))

        model.fit(dataset,
                  steps_per_epoch=len(batch_generator),
                  epochs=5,
                  verbose=1,
                  max_queue_size=10,
                  workers=3)

    def test_single_gpu_float16_no_masking_no_dropout_noise_shape_no_sample_weight_mode(self):
        dtype = self._enable_float16()

        batch_generator = FakeChatGenerator(
            num_unique_symbols=self.num_unique_symbols,
            max_seq_length=self.max_seq_length,
            batch_size=self.batch_size,
            return_sample_weights=False,
            dtype=dtype)

        model_generator = Seq2SeqWithSubmodels(TestIssuesTFKeras.KERAS_CLASSES,
                                               self.max_seq_length,
                                               self.num_unique_symbols,
                                               use_masking=False,
                                               disable_dropout_noise_shape=True,
                                               dtype=dtype)

        def create_optimizer():
            # optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
            optimizer = Adam(lr=1e-4)
            return tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

        with tf.device('/gpu:0'):
            models = model_generator.stamp_train_model()
            model = models["train_model"]

        self._compile_model(model, create_optimizer, sample_weight_mode=False)

        # TODO : shapes are only correct when using one-hot encoding
        dataset = tf.data.Dataset.from_generator(
            batch_generator,
            ({
                 'chat-input': dtype,
                 'teacher-forcing-input': dtype
             },

             dtype),  # outputs
            ({
                 'chat-input': (None, None, self.num_unique_symbols),
                 'teacher-forcing-input': (None, None, self.num_unique_symbols)
             },
             (None, None, self.num_unique_symbols)))

        model.fit(dataset,
                  steps_per_epoch=len(batch_generator),
                  epochs=5,
                  verbose=1,
                  max_queue_size=10,
                  workers=3)

    def test_multi_gpu_float16_no_masking_no_dropout_noise_shape_no_sample_weight_mode(self):
        dtype = self._enable_float16()

        batch_generator = FakeChatGenerator(
            num_unique_symbols=self.num_unique_symbols,
            max_seq_length=self.max_seq_length,
            batch_size=self.batch_size,
            return_sample_weights=False,
            dtype=dtype)

        model_generator = Seq2SeqWithSubmodels(TestIssuesTFKeras.KERAS_CLASSES,
                                               self.max_seq_length,
                                               self.num_unique_symbols,
                                               use_masking=False,
                                               disable_dropout_noise_shape=True,
                                               dtype=dtype)

        def create_optimizer():
            # optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
            optimizer = Adam(lr=1e-4)
            return tf.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)

        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            models = model_generator.stamp_train_model()
            model = models["train_model"]

            self._compile_model(model, create_optimizer, sample_weight_mode=False)

        # TODO : shapes are only correct when using one-hot encoding
        dataset = tf.data.Dataset.from_generator(
            batch_generator,
            ({
                 'chat-input': dtype,
                 'teacher-forcing-input': dtype
             },

             dtype),  # outputs
            ({
                 'chat-input': (None, None, self.num_unique_symbols),
                 'teacher-forcing-input': (None, None, self.num_unique_symbols)
             },
             (None, None, self.num_unique_symbols)))

        model.fit(dataset,
                  steps_per_epoch=len(batch_generator),
                  epochs=5,
                  verbose=1,
                  max_queue_size=10,
                  workers=3)

    def _compile_model(self, model, create_optimizer, sample_weight_mode=True):

        optimizer = create_optimizer()

        if sample_weight_mode:
            model.compile(loss=self.loss_type,
                          sample_weight_mode="temporal",
                          optimizer=optimizer,
                          weighted_metrics=["accuracy"])
        else:
            model.compile(loss=self.loss_type,
                          optimizer=optimizer,
                          metrics=["accuracy"])

    def _enable_float32(self):
        dtype = 'float32'

        tf.compat.v1.keras.backend.set_floatx(dtype)
        tf.compat.v1.keras.backend.set_epsilon(1e-7)

        return dtype

    def _enable_float16(self):
        dtype = 'float16'

        tf.compat.v1.keras.backend.set_floatx(dtype)
        tf.compat.v1.keras.backend.set_epsilon(1e-4)

        return dtype


if __name__ == '__main__':
    unittest.main()