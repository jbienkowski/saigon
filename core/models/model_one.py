from base.model_base import ModelBase
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout


class ModelOne(ModelBase):
    def __init__(self):
        ModelBase.__init__(self, "model_one")

    def define_model(self):
        self.model = Sequential()
        self.model.add(Dropout(0.2, seed=42, input_shape=(self.total_inputs,)))
        self.model.add(Dense(128, activation="relu", name="m3_hidden1"))
        self.model.add(Dropout(0.25, seed=42))
        self.model.add(Dense(64, activation="relu", name="m3_hidden2"))
        self.model.add(Dense(15, activation="relu", name="m3_hidden3"))
        self.model.add(Dense(10, activation="softmax", name="m3_output"))
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
