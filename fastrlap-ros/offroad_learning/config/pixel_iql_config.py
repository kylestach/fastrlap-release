import ml_collections
from ml_collections.config_dict import config_dict


def get_config():
    config = ml_collections.ConfigDict()

    config.model_cls = "PixelIQLLearner"

    config.actor_lr = 3e-4
    config.value_lr = 3e-4
    config.critic_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.cnn_features = (32, 32, 32, 32)
    config.cnn_filters = (3, 3, 3, 3)
    config.cnn_strides = (2, 2, 2, 2)
    config.cnn_padding = "VALID"
    config.latent_dim = 50
    config.encoder = "d4pg"

    config.discount = 0.99

    config.num_qs = 10
    config.num_min_qs = 2
    config.num_vs = 1

    config.critic_layer_norm = True
    config.value_layer_norm = True

    config.expectile = 0.9  # The actual tau for expectiles.
    config.temperature = 3.0
    config.cosine_decay = True

    config.tau = 0.005

    return config
