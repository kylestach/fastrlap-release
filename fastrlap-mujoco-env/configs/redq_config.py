import ml_collections
from ml_collections.config_dict import config_dict


def get_config():
    config = ml_collections.ConfigDict()

    config.model_cls = "SACLearner"

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 1e-4

    config.hidden_dims = (256, 256)

    config.discount = 0.997

    config.num_qs = 10
    config.num_min_qs = 2

    # config.critic_dropout_rate = 0.01
    config.critic_layer_norm = True

    config.tau = 0.005
    config.init_temperature = 1.0
    config.target_entropy = -1

    return config
