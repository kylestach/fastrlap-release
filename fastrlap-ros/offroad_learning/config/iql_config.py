import ml_collections
from ml_collections.config_dict import config_dict


def get_config():
    config = ml_collections.ConfigDict()

    config.model_cls = 'IQLLearner'

    config.actor_lr = 1e-3
    config.value_lr = 3e-4
    config.critic_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.discount = 0.99

    config.num_qs = 2

    config.expectile = 0.7  # The actual tau for expectiles.
    config.temperature = 3.0
    config.cosine_decay = True

    config.actor_weight_decay = config_dict.placeholder(float)
    config.critic_weight_decay = config_dict.placeholder(float)
    config.value_weight_decay = config_dict.placeholder(float)

    config.tau = 0.005  # For soft target updates.

    return config
