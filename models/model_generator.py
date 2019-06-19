import abc

from basics.base import Base


class ModelGenerator(Base, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def stamp_train_model(self, batch_size, model_weights=None, disable_training=None):
        self._log.error('This method must be implemented in a child method')
        return None

    @abc.abstractmethod
    def stamp_infer_model(self, batch_size, model_weights=None):
        self._log.error('This method must be implemented in a child method')
        return None
