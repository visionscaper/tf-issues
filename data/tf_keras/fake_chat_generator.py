from data.keras.fake_chat_generator import FakeChatGenerator as KerasFakeChatGenerator


class FakeChatGenerator(KerasFakeChatGenerator):

    def __next__(self):
        with self.lock:

            input_chats = self._fake_chats()
            output_chats = self._fake_chats()

            encoded_input = self._encode_input(input_chats)
            encoded_teacher_forcing_input = self._encode_output_feedback(output_chats)
            encoded_output, sample_weights = self._encode_output(output_chats)

            model_inputs = {
                "chat-input": encoded_input,
                "teacher-forcing-input": encoded_teacher_forcing_input
            }

            if self.return_sample_weights:
                return model_inputs, encoded_output, sample_weights
            else:
                return model_inputs, encoded_output

    def __call__(self, *args, **kwargs):
        return self
