import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Masking, Dense, Activation, Dropout
from tensorflow.keras.layers import TimeDistributed

from tensorflow.keras.layers import GRU

from models.model_generator import ModelGenerator

import basics.base_utils as _

from utils import select_value

_DEFAULT_EMBEDDING_SIZE = 16
_DEFAULT_ENCODER_STATE_SIZE = [1024, 1024, 1024]
_DEFAULT_DECODER_STATE_SIZE = [1024, 1024, 1024]
_DEFAULT_DECODER_OUTPUT_MAPPING_SIZE = 512

_DISABLE_TRAINING_DEFAULTS = {
    "embedding_model": False,
    "encoder_model": False,
    "decoder_model": False
}


class Seq2SeqWithSubmodels(ModelGenerator):

    def __init__(self,
                 max_seq_length,
                 num_symbols,
                 encoder_state_size=None,
                 rnn_layer=None,
                 embedding_size=_DEFAULT_EMBEDDING_SIZE,
                 decoder_state_size=None,
                 decoder_output_mapping_size=None,
                 dense_output_activation='tanh',
                 output_activation='softmax',
                 drop_out=0.05,
                 use_masking=True,
                 dtype='float16',
                 use_partially_known_dropout_noise_shape=False,
                 **kwargs):

        super(Seq2SeqWithSubmodels, self).__init__()

        # For encoder input : Sequence + CHAT_END ==> seq. length + 1
        # For decoder feedback input : CHAT_START + Sequence + CHAT_END ==> seq. length + 2
        # For decoder output : Sequence + CHAT_END + (optional) CONVERSATION_END ==> seq. length + 2
        #
        # Common length is seq. length + 2
        self.max_seq_length = max_seq_length + 2

        self.num_symbols = num_symbols

        self.rnn_layer = rnn_layer or GRU

        self.drop_out = drop_out
        self.use_masking = use_masking
        self.masking_value = None
        if self.use_masking:
            self._log.debug("Using masking.")
            self.masking_value = 0.0

        self.dtype = dtype
        self._log.info('Using data type : %s' % self.dtype)

        self.dense_output_activation = dense_output_activation
        self.output_activation = output_activation

        self.embedding_size = embedding_size
        self.encoder_state_size = select_value(encoder_state_size, _DEFAULT_ENCODER_STATE_SIZE)
        self.decoder_state_size = select_value(decoder_state_size, _DEFAULT_DECODER_STATE_SIZE)
        self.decoder_output_mapping_size = select_value(decoder_output_mapping_size,
                                                        _DEFAULT_DECODER_OUTPUT_MAPPING_SIZE)

        self.use_partially_known_dropout_noise_shape = use_partially_known_dropout_noise_shape
        self._log.info('Using partially unknown dropout noise shape : %s' % self.use_partially_known_dropout_noise_shape)

    def stamp_train_model(self, batch_size=None, checkpoint=None, disable_training=None):
        self._log.info('Stamp train model ...')

        for_inference = False

        if not disable_training:
            disable_training = _DISABLE_TRAINING_DEFAULTS

        input_batch_shape = (batch_size, self.max_seq_length, self.num_symbols)

        teacher_forcing_batch_shape = (batch_size,
                                       self.max_seq_length if not for_inference else 1,
                                       self.num_symbols)

        chat_input = tf.keras.Input(
            name="chat_input",
            batch_shape=input_batch_shape,
            dtype=self.dtype)

        teacher_forcing_input = tf.keras.Input(
            name="teacher-forcing-input",
            batch_shape=teacher_forcing_batch_shape,
            dtype=self.dtype)

        embedding_model = self._create_embedding_model(batch_size)
        encoder_model = self._create_encoder_model(embedding_model, batch_size)
        decoder_model = self._create_decoder_model(embedding_model, batch_size, for_inference)

        thought = encoder_model(chat_input)
        output = decoder_model([thought, teacher_forcing_input])

        train_model = Model(inputs=[input, teacher_forcing_input],
                            outputs=output)

        if checkpoint is not None:
            try:
                self._log.debug("Loading checkpoint : [%s]" % checkpoint)
                train_model.load_weights(checkpoint)
            except Exception as e:
                _.log_exception(self._log, "Unable to set weights for stamped model", e)
                return None

        self._log.debug('Embedding model :')
        embedding_model.summary()
        self._log.debug('Encoder model :')
        encoder_model.summary()
        self._log.debug('Decoder model :')
        decoder_model.summary()
        self._log.debug('Training model :')
        train_model.summary()

        models = {
            "train_model": train_model,
            "embedding_model": embedding_model,
            "encoder_model": encoder_model,
            "decoder_model": decoder_model
        }

        self._set_model_trainability(models, disable_training)

        return models

    def stamp_infer_model(self, batch_size, model_weights=None):
        self._log.error('This method is not implemented')
        return None

    def _create_embedding_layer(self):
        # Use simple Dense layer for character level embedding
        return TimeDistributed(Dense(self.embedding_size), name="embedding-layer")

    def _create_encoder_layers(self):
        num_encoder_layers = len(self.encoder_state_size)

        layers = []

        for layer_idx in range(num_encoder_layers):
            is_last = layer_idx == (num_encoder_layers - 1)

            layer = self.rnn_layer(
                self.encoder_state_size[layer_idx],
                name="encoder-layer-%d" % layer_idx,
                stateful=False,
                return_state=False,  # output and state are the same for GRU
                return_sequences=not is_last)

            layers.append(layer)

        return layers

    def _create_decoder_layers(self, for_inference=False):
        num_decoder_layers = len(self.decoder_state_size)

        layers = []
        for layer_idx in range(num_decoder_layers):
            layers.append(self.rnn_layer(
                self.decoder_state_size[layer_idx],
                name="decoder-layer-%d" % layer_idx,
                stateful=for_inference,
                return_state=False,  # output and state are the same for GRU
                return_sequences=True))

        return layers

    def _create_dense_output(self, input, layers):
        output = input
        for layer in layers:
            output = layer(output)
            output = Activation(self.dense_output_activation)(output)

            noise_shape = (None, 1, None)
            if not self.use_partially_known_dropout_noise_shape:
                shape = tf.shape(output)
                noise_shape = (shape[0], 1, shape[2])

            output = Dropout(self.drop_out, noise_shape=noise_shape)(output)

        return output

    def _create_rnn_output(self, input, layers, initial_state=None):
        """

        NOTE: For GRU, output and state are the same value

        :param input:
        :param layers:
        :param initial_state:
        :return:
        """

        output = input
        for idx, layer in enumerate(layers):
            init_state = initial_state if idx == 0 else None

            output = layer(output, initial_state=init_state)

            noise_shape = (None, 1, None)
            if not self.use_partially_known_dropout_noise_shape:
                shape = tf.shape(output)
                noise_shape = (shape[0], 1, shape[2])

            output = Dropout(self.drop_out, noise_shape=noise_shape)(output)

        return output

    def _create_encoder_output(self, encoder_input, encoder_layers):
        return self._create_rnn_output(encoder_input, encoder_layers)

    def _create_decoder_output(self, feedback_decoder_input, initial_state, decoder_layers):

        decoder_output = self._create_rnn_output(feedback_decoder_input,
                                                 decoder_layers,
                                                 initial_state=initial_state)

        if not (self.decoder_output_mapping_size is None):
            self._log.info(f'Using decoder output mapping layer with size : {self.decoder_output_mapping_size}')

            output_mapping_layer = TimeDistributed(Dense(self.decoder_output_mapping_size),
                                                   name="decoder-output-mapping-layer")
            decoder_output = self._create_dense_output(decoder_output,
                                                       [output_mapping_layer])

        decoder_output = TimeDistributed(
            Dense(
                self.num_symbols,
                activation=self.output_activation
            ),
            name="decoder-output-layer")(decoder_output)

        return decoder_output

    def _create_embedding_model(self, batch_size=1):
        self._log.debug('Creating embedding model ...')
        embedding_layer = self._create_embedding_layer()

        embedding_input = tf.keras.Input(
            name="encoder-input",
            batch_shape=(batch_size, self.max_seq_length, self.num_symbols),
            dtype=self.dtype)

        masked_input = embedding_input
        if self.use_masking:
            self._log.debug('Adding masking layer ...')
            masked_input = Masking()(embedding_input)

        embedding_output = embedding_layer(masked_input)

        return Model(inputs=[embedding_input],
                     outputs=[embedding_output],
                     name='embedding-model')

    def _create_encoder_model(self, embedding_model, batch_size=1):
        self._log.debug('Creating encoder model ...')
        encoder_layers = self._create_encoder_layers()

        input_batch_shape = (batch_size, self.max_seq_length, self.num_symbols)

        encoder_input = tf.keras.Input(
            name="chat-encoder-input",
            batch_shape=input_batch_shape,
            dtype=self.dtype)

        embedded_input = embedding_model(encoder_input)

        output = self._create_encoder_output(embedded_input, encoder_layers)

        return Model(inputs=[encoder_input],
                     outputs=[output],
                     name='chat-encoder-model')

    def _create_decoder_model(self, embedding_model, batch_size=1, for_inference=False):
        self._log.debug('Creating decoder model with agent names...')

        decoder_layers = self._create_decoder_layers(for_inference)

        teacher_forcing_batch_shape = (batch_size,
                                       self.max_seq_length if not for_inference else 1,
                                       self.num_symbols)

        teacher_forcing_decoder_input = tf.keras.Input(
            name="teacher-forcing-decoder-input",
            batch_shape=teacher_forcing_batch_shape,
            dtype=self.dtype)

        embedded_teacher_forcing_input = embedding_model(teacher_forcing_decoder_input)

        # In inference mode, the state is not a model input but set as state directly on the first layer of the
        # decoder model
        if not for_inference:

            thought_decoder_input = tf.keras.Input(
                name="thought-decoder-input",
                batch_shape=(batch_size, self.encoder_state_size[-1]),
                dtype=self.dtype)

            decoder_output = self._create_decoder_output(embedded_teacher_forcing_input,
                                                         initial_state=thought_decoder_input,
                                                         decoder_layers=decoder_layers)

            return Model(inputs=[thought_decoder_input,
                                 teacher_forcing_decoder_input],
                         outputs=decoder_output,
                         name='decoder-model')
        else:
            # Initial state set during inference
            decoder_output = self._create_decoder_output(embedded_teacher_forcing_input,
                                                         initial_state=None,
                                                         decoder_layers=decoder_layers)

            return Model(inputs=[teacher_forcing_decoder_input],
                         outputs=decoder_output,
                         name='decoder-model')

    def _set_model_trainability(self, models, disable_training):
        if type(disable_training) != dict:
            self._log.error(f'disable_training parameter is not a dict, unable to set model trainability ...')
            return

        for model_name, disable_model_training in disable_training.items():
            if disable_model_training:
                if model_name in models:
                    self._log.debug(f'Disabling training for model {model_name} ...')
                    models[model_name].trainable = False
                else:
                    self._log.error(f'Model {model_name} is unknown, unable to disabling training for this model ...')
