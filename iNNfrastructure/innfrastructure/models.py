from . import config as cfg

import os
import tensorflow.keras as keras

models = {}


def get_model(model_type: str) -> keras.models.Sequential:
    """Load a model with some caching."""
    if model_type not in models.keys():
        models[model_type] = keras.models.load_model(
            os.path.join(cfg.MODEL_DIR, f"{model_type}.h5")
        )
        # No error handling is necessary here,
        # as we just use the FileNotFoundError from load_model
        models[model_type].compile()
    return models[model_type]
