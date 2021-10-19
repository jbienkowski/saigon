from .base.model_base import ModelBase
from keras.models import Model
from keras import layers


class ModelOne(ModelBase):
    def __init__(self, cfg, **kwargs):
        ModelBase.__init__(self, cfg=cfg, model_name="model_one", **kwargs)
        self._define_model()

    def _define_model(self):
        input_tensor = layers.Input(shape=(self.total_inputs,1))
        drop_out_rate = 0.1

        x = layers.Conv1D(8, 11, padding='valid', activation='relu', strides=1)(input_tensor)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(drop_out_rate)(x)
        x = layers.Conv1D(16, 7, padding='valid', activation='relu', strides=1)(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(drop_out_rate)(x)
        x = layers.Conv1D(32, 5, padding='valid', activation='relu', strides=1)(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(drop_out_rate)(x)
        x = layers.Conv1D(64, 5, padding='valid', activation='relu', strides=1)(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Dropout(drop_out_rate)(x)
        x = layers.Conv1D(128, 3, padding='valid', activation='relu', strides=1)(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(drop_out_rate)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(drop_out_rate)(x)

        output_tensor = layers.Dense(2, activation='softmax')(x)

        self.model = Model(inputs=input_tensor, outputs=output_tensor)

        self.model.compile(
            optimizer="Adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
        )
