from random import randint

from threading import Lock

import numpy as np

from keras.utils import np_utils

from basics.base import Base


class FakeChatGenerator(Base):

    def __init__(
            self,
            num_unique_symbols,
            max_seq_length,
            batch_size,
            return_sample_weights=True,
            dtype='float16',
            **kwargs):

        super().__init__(**kwargs)

        # Unknown symbols have index 1
        self.chat_start_index = 2
        self.chat_end_index = 3
        self.num_unique_symbols = num_unique_symbols
        self.max_seq_length = max_seq_length

        self.batch_size = batch_size

        self.return_sample_weights = return_sample_weights

        self.dtype = dtype

        self.lock = Lock()

        self._log.info('Using data type : %s' % self.dtype)

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:

            input_chats = self._fake_chats()
            output_chats = self._fake_chats()

            encoded_input = self._encode_input(input_chats)
            encoded_teacher_forcing_input = self._encode_output_feedback(output_chats)
            encoded_output, sample_weights = self._encode_output(output_chats)

            model_inputs = [encoded_input, encoded_teacher_forcing_input]

            if self.return_sample_weights:
                return model_inputs, encoded_output, sample_weights
            else:
                return model_inputs, encoded_output

    def __len__(self):
        return 20

    def _fake_chats(self):
        return [self._fake_chat() for _ in range(self.batch_size)]

    def _fake_chat(self):
        # Unknown symbols are at index 1
        chat_length = randint(2, self.max_seq_length)
        chat = np.around(4+np.random.rand(chat_length))
        return chat

    def _encode_input(self, chats):
        input_batch = np.zeros((self.batch_size, (self.max_seq_length + 2), self.num_unique_symbols),
                               dtype=self.dtype)

        for i, chat in enumerate(chats):
            # -1 because symbol index mapping starts at one
            _chat = chat + [self.chat_end_index]
            _chat = self._encode_chat(_chat)

            input_batch[i, 0:len(_chat), :] = _chat

        return input_batch

    def _prepare_output(self, chat):
        # -1 because symbol index mapping starts at one
        chat = [self.chat_start_index] + chat + [self.chat_end_index]

        return chat

    def _encode_output_feedback(self, chats):
        batch = np.zeros((self.batch_size, self.max_seq_length + 2, self.num_unique_symbols), dtype=self.dtype)

        for i, chat in enumerate(chats):
            chat = self._prepare_output(chat)

            # start symbol is first input
            chat = chat[0:len(chat) - 1]

            _chat = self._encode_chat(chat)
            batch[i, 0:len(chat), :] = _chat

        return batch

    def _encode_output(self, chats):
        batch = np.zeros((self.batch_size, self.max_seq_length + 2, self.num_unique_symbols), dtype=self.dtype)
        sample_weights_batch = np.zeros((self.batch_size, self.max_seq_length + 2), dtype=self.dtype)

        for i, chat in enumerate(chats):
            chat = self._prepare_output(chat)

            # predict one character ahead after start symbol
            chat = chat[1:len(chat)]

            _chat = self._encode_chat(chat)
            batch[i, 0:len(chat), :] = _chat

            for j, ci in enumerate(chat):
                sample_weights_batch[i, j] = 1

        return batch, sample_weights_batch

    def _encode_chat(self, chat):
        # Unknown symbols are at index 1
        chat = np.array(chat) - 1
        return np_utils.to_categorical(chat, self.num_unique_symbols)
