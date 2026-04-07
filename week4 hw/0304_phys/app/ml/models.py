"""
ModelFactory — Keras model construction for MOD-02, MOD-04, MOD-05.
No Qt imports. All models returned pre-compiled.
SDD-PHYSAI-003 §3.2
"""


class ModelFactory:

    @staticmethod
    def projectile_regression(learning_rate: float = 0.001):
        """
        MOD-02: (v₀, θ, t) → (x, y)
        Input(3) → Dense(128,relu)+DO(0.1) → Dense(64,relu)+DO(0.1) → Dense(32,relu)+DO(0.1) → Dense(2)
        """
        from tensorflow import keras
        model = keras.Sequential([
            keras.layers.Input(shape=(3,)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(2, activation='linear'),
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                      loss='mse', metrics=['mae'])
        return model

    @staticmethod
    def pendulum_period(learning_rate: float = 0.001):
        """
        MOD-04: (L, θ₀) → T
        Input(2) → Dense(64,relu)+DO(0.1) → Dense(32,relu)+DO(0.1) → Dense(16,relu) → Dense(1)
        """
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
        model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                      loss='mse', metrics=['mae', 'mape'])
        return model

    @staticmethod
    def air_resistance_range(learning_rate: float = 0.001):
        """
        MOD-05: (v₀, θ) → R
        Input(2) → Dense(64,relu) → Dense(64,relu) → Dense(32,relu) → Dense(1)
        """
        from tensorflow import keras
        model = keras.Sequential([
            keras.layers.Input(shape=(2,)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='linear'),
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                      loss='mse', metrics=['mae'])
        return model
