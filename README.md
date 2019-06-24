# tf-issues
This repo contains unit tests to contrast the functioning of `Keras` against `tf.keras`.

## Running the tests

To run the `Keras` and `tf.keras` tests you need to set up different virtual environments, both based on Python 3.6.

For the virtual environment to run the `Keras` tests,  please install the requirements in `requirements_keras_env.txt`.
For the `tf.keras` test install the `requirements_tensorflow_keras_env.txt`.

Assume we have created virtual environments `~/.virtualenvs/keras` and `/.virtualenvs/tf-keras`.

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



  
