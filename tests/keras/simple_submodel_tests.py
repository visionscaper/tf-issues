import unittest

import tensorflow as tf

from keras import backend as K

from keras.models import Model
from keras.layers import Input, Dense, Dropout, TimeDistributed

from keras.layers.cudnn_recurrent import CuDNNGRU

from keras.optimizers import Adam

from data.keras.fake_chat_generator import FakeChatGenerator

# import pydevd_pycharm
# pydevd_pycharm.settrace('192.168.178.8', port=57491, stdoutToServer=True, stderrToServer=True)


class SimpleSubmodelTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
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

        sess = tf.Session(config=SimpleSubmodelTests.TF_CONFIG)
        K.set_session(sess)

    def test_sub_models(self):
        dtype = self._enable_float32()

        batch_generator = FakeChatGenerator(
            num_unique_symbols=self.num_unique_symbols,
            max_seq_length=self.max_seq_length,
            batch_size=self.batch_size,
            return_sample_weights=True,
            dtype=dtype)

        with tf.device('/gpu:0'):
            bot_model = self._build_simple_seq2seq_bot()

        self._compile_model(bot_model, lambda: Adam(lr=1e-4))

        bot_model.fit_generator(batch_generator,
                                steps_per_epoch=len(batch_generator),
                                epochs=5,
                                verbose=1,
                                max_queue_size=10,
                                workers=3)

    def test_sub_models_with_embedding_model(self):
        dtype = self._enable_float32()

        batch_generator = FakeChatGenerator(
            num_unique_symbols=self.num_unique_symbols,
            max_seq_length=self.max_seq_length,
            batch_size=self.batch_size,
            return_sample_weights=True,
            dtype=dtype)

        with tf.device('/gpu:0'):
            bot_model = self._build_simple_seq2seq_bot_with_embedding_model()

        self._compile_model(bot_model, lambda: Adam(lr=1e-4))

        bot_model.fit_generator(batch_generator,
                                steps_per_epoch=len(batch_generator),
                                epochs=5,
                                verbose=1,
                                max_queue_size=10,
                                workers=3)

    def test_sub_models_with_embedding_model_inside_encoder_and_decoder(self):
        dtype = self._enable_float32()

        batch_generator = FakeChatGenerator(
            num_unique_symbols=self.num_unique_symbols,
            max_seq_length=self.max_seq_length,
            batch_size=self.batch_size,
            return_sample_weights=True,
            dtype=dtype)

        with tf.device('/gpu:0'):
            bot_model = self._build_simple_seq2seq_bot_with_embedding_model_inside_encoder_and_decoder()

        self._compile_model(bot_model, lambda: Adam(lr=1e-4))

        bot_model.fit_generator(batch_generator,
                                steps_per_epoch=len(batch_generator),
                                epochs=5,
                                verbose=1,
                                max_queue_size=10,
                                workers=3)

    def _build_simple_seq2seq_bot(self):
        state_size = 1024

        encoder = self._build_encoder(state_size)
        decoder = self._build_decoder(state_size)

        user_inp = Input(batch_shape=(None, self.max_seq_length+2, self.num_unique_symbols),
                         name='user-input')

        teacher_inp = Input(batch_shape=(None, self.max_seq_length+2, self.num_unique_symbols),
                            name='teacher-input')

        thought = encoder(user_inp)
        response = decoder([teacher_inp, thought])

        return Model(inputs=[user_inp, teacher_inp], outputs=[response], name='bot-model')

    def _build_simple_seq2seq_bot_with_embedding_model(self):
        state_size = 1024

        embedder = self._build_embedder()
        encoder = self._build_encoder(state_size)
        decoder = self._build_decoder(state_size)

        user_inp = Input(batch_shape=(None, self.max_seq_length+2, self.num_unique_symbols),
                         name='user-input')

        teacher_inp = Input(batch_shape=(None, self.max_seq_length+2, self.num_unique_symbols),
                            name='teacher-input')

        embedded_user_inp = embedder(user_inp)
        thought = encoder(embedded_user_inp)

        embedded_teacher_inp = embedder(teacher_inp)
        response = decoder([embedded_teacher_inp, thought])

        return Model(inputs=[user_inp, teacher_inp], outputs=[response], name='bot-model')

    def _build_simple_seq2seq_bot_with_embedding_model_inside_encoder_and_decoder(self):
        state_size = 1024

        embedder = self._build_embedder()
        encoder = self._build_encoder(state_size, embedding_model=embedder)
        decoder = self._build_decoder(state_size, embedding_model=embedder)

        user_inp = Input(batch_shape=(None, self.max_seq_length+2, self.num_unique_symbols),
                         name='user-input')

        teacher_inp = Input(batch_shape=(None, self.max_seq_length+2, self.num_unique_symbols),
                            name='teacher-input')

        thought = encoder(user_inp)
        response = decoder([teacher_inp, thought])

        return Model(inputs=[user_inp, teacher_inp], outputs=[response], name='bot-model')

    def _build_embedder(self):
        # NOTE: self.max_seq_length+2 is used because of added start and end symbols

        embedding_in = Input(batch_shape=(None, self.max_seq_length+2, self.num_unique_symbols),
                             name='encoder-input')

        # Use simple Dense layer for character level embedding
        embedded_in = TimeDistributed(Dense(self.embedding_size), name="embedding-layer")(embedding_in)

        return Model(inputs=[embedding_in],
                     outputs=[embedded_in],
                     name='embedding-model')

    def _build_encoder(self, state_size, add_dropout=True, embedding_model=None):
        # NOTE: self.max_seq_length+2 is used because of added start and end symbols

        encoder_in = Input(batch_shape=(None, self.max_seq_length+2, self.num_unique_symbols),
                           name='encoder-input')

        embedded_in = encoder_in
        if embedding_model:
            embedded_in = embedding_model(encoder_in)

        out = CuDNNGRU(state_size)(embedded_in)

        if add_dropout:
            out = Dropout(0.1, name='encoder-dropout')(out)

        return Model(inputs=[encoder_in],
                     outputs=[out],
                     name='encoder-model')

    def _build_decoder(self, state_size, add_dropout=True, embedding_model=None):
        # NOTE: self.max_seq_length+2 is used because of added start and end symbols

        feedback_in = Input(batch_shape=(None, self.max_seq_length+2, self.num_unique_symbols),
                            name='decoder-feedback-input')

        thought_in = Input(batch_shape=(None, state_size),
                           name='decoder-thought-input')

        embedded_feedback_in = feedback_in
        if embedding_model:
            embedded_feedback_in = embedding_model(feedback_in)

        out = CuDNNGRU(state_size, return_sequences=True)(embedded_feedback_in, initial_state=thought_in)

        if add_dropout:
            out = Dropout(0.1, noise_shape=(None, 1, None), name='encoder-dropout')(out)

        out = TimeDistributed(Dense(self.num_unique_symbols,
                                    activation='softmax',
                                    name='decoder-output-layer'))(out)

        return Model(inputs=[feedback_in, thought_in],
                     outputs=[out],
                     name='decoder-model')

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
