import ml_collections
from ml_collections.config_dict import config_dict


def get_config():
    config = ml_collections.ConfigDict()

    config.model_cls = "DrQLearner"

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 1e-4

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

    config.critic_layer_norm = True

    config.tau = 0.005
    config.init_temperature = 0.5
    config.target_entropy = -2.0
    config.backup_entropy = False

    config.freeze_encoder = True

    return config
