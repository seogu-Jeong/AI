"""
ModelFactory — Keras model construction for MOD-01 and MOD-03.
No Qt imports. All models returned pre-compiled.
"""
from typing import List


class ModelFactory:

    # FR-MOD01-01: four architecture presets
    ARCH_OPTIONS = {
        'Small [32]':            [32],
        'Medium [64, 64]':       [64, 64],
        'Large [128, 128]':      [128, 128],
        'XLarge [128, 128, 64]': [128, 128, 64],
    }

    @staticmethod
    def count_params(hidden_layers: List[int], activation: str = 'tanh') -> int:
        """Returns approximate parameter count for display (FR-MOD01-03)."""
        from tensorflow import keras
        m = ModelFactory.function_approximator(hidden_layers, activation, 0.01)
        return int(m.count_params())

    @staticmethod
    def function_approximator(
        hidden_layers: List[int],
        activation: str = 'tanh',
        learning_rate: float = 0.01,
    ):
        """MOD-01: 1-D input → 1-D output regression."""
        from tensorflow import keras
        layers = [keras.layers.Input(shape=(1,))]
        for units in hidden_layers:
            layers.append(keras.layers.Dense(units, activation=activation))
        layers.append(keras.layers.Dense(1, activation='linear'))
        model = keras.Sequential(layers)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae'],
        )
        return model

    @staticmethod
    def overfitting_suite():
        """
        MOD-03: three models per SRS-PHYSAI-002 §4.5.
        Underfit: Dense(4), Good: Dense(32)+DO(0.2)+Dense(16)+DO(0.2),
        Overfit: Dense(256,128,64,32) — no regularisation.
        """
        from tensorflow import keras

        def _compile(m):
            m.compile(optimizer='adam', loss='mse', metrics=['mae'])
            return m

        return {
            'underfit': _compile(keras.Sequential([
                keras.layers.Input(shape=(1,)),
                keras.layers.Dense(4, activation='relu'),
                keras.layers.Dense(1),
            ])),
            'good': _compile(keras.Sequential([
                keras.layers.Input(shape=(1,)),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(16, activation='relu'),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(1),
            ])),
            'overfit': _compile(keras.Sequential([
                keras.layers.Input(shape=(1,)),
                keras.layers.Dense(256, activation='relu'),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dense(64,  activation='relu'),
                keras.layers.Dense(32,  activation='relu'),
                keras.layers.Dense(1),
            ])),
        }
