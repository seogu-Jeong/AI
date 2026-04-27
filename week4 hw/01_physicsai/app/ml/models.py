"""
ModelFactory — Keras model construction. No Qt imports.
All models returned pre-compiled and ready for training.
"""
from typing import List


class ModelFactory:

    @staticmethod
    def function_approximator(
        hidden_layers: List[int],
        activation: str = 'tanh',
        learning_rate: float = 0.01,
    ):
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
    def projectile_regression(learning_rate: float = 0.001):
        from tensorflow import keras
        model = keras.Sequential([
            keras.layers.Input(shape=(3,)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(2, activation='linear'),
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss='mse', metrics=['mae'],
        )
        return model

    @staticmethod
    def overfitting_suite():
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
                keras.layers.Dense(64, activation='relu'),
                keras.layers.Dense(32, activation='relu'),
                keras.layers.Dense(1),
            ])),
        }

    @staticmethod
    def pendulum_period(learning_rate: float = 0.001):
        from tensorflow import keras
        model = keras.Sequential([
            keras.layers.Input(shape=(2,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(1, activation='linear'),
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss='mse', metrics=['mae', 'mape'],
        )
        return model

    @staticmethod
    def air_resistance_range(learning_rate: float = 0.001):
        from tensorflow import keras
        model = keras.Sequential([
            keras.layers.Input(shape=(2,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='linear'),
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate),
            loss='mse', metrics=['mae'],
        )
        return model
