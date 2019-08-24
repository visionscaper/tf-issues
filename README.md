# tf-issues
This repo contains unit tests to contrast the functioning of `Keras` against `tf.keras`.

## Special requirements

The Tensorflow 2.0 Keras environment (`tf20-keras`, see below), requires at least CuDNN 7.6.0 installed
(assuming `tf-nightly-gpu-2.0-preview 2.0.0.dev20190824`)

## Running the tests

To run the `Keras` and `tf.keras` tests you need to set up different virtual environments, both based on Python 3.6.

For the virtual environment to run the `Keras` tests,  please install the requirements in `requirements_keras_env.txt`.
For the `tf.keras` test install the `requirements_tensorflow_keras_env.txt`.
For the `tf20.keras` test , based on Tensorflow 2.0, install the `requirements_tensorflow20_keras_env.txt`.

Assume we have created virtual environments `~/.virtualenvs/keras`, `/.virtualenvs/tf-keras` and `/.virtualenvs/tf20-keras`.

### Running the `Keras` tests
```
source ~/.virtualenvs/keras/bin/activate
python3 tests/keras/tests.py
```

### Running the `tf.keras` tests
```
source ~/.virtualenvs/tf-keras/bin/activate
python3 tests/tf_keras/tests.py
```

### Running the `tf20.keras` tests
```
source ~/.virtualenvs/tf20-keras/bin/activate
python3 tests/tf20_keras/tests.py
```

#### Warning
25 August 2019:
The tests related to float16 (at least for `tf20.keras`) result in segmentation faults.

Run a individual test as follows:

```python
python3 tests/tf20_keras/tests.py TestIssuesTFKeras.test_multi_gpu_float32_no_masking_no_dropout_noise_shape_sample_weight_mode
```
