from collections import defaultdict
from tensor_dict_msgs.msg import TensorDict, Tensor
import numpy as np


def to_ros_msg(tensor_dict, msg=None, prefix=None):
    def format_key(key):
        if prefix is None:
            return key
        else:
            return f"{prefix}/{key}"

    if msg is None:
        msg = TensorDict()

    for key, value in tensor_dict.items():
        if isinstance(value, np.ndarray):
            tensor_msg = Tensor()
            tensor_msg.name = format_key(key)

            if value.dtype == np.uint8:
                tensor_msg.uint8_data = value.flatten().tolist()
            elif value.dtype == np.float32:
                tensor_msg.float32_data = value.flatten().tolist()
            elif value.dtype == np.float64:
                tensor_msg.float64_data = value.flatten().tolist()
            elif value.dtype == np.int32:
                tensor_msg.int32_data = value.flatten().tolist()
            elif value.dtype == np.int64:
                tensor_msg.int64_data = value.flatten().tolist()
            elif value.dtype == np.bool_:
                tensor_msg.bool_data = value.flatten().tolist()
            else:
                raise ValueError(f"Unsupported dtype {value.dtype}")

            tensor_msg.shape = list(value.shape)
            msg.tensors.append(tensor_msg)
        elif isinstance(value, dict):
            to_ros_msg(value, msg=msg, prefix=format_key(key))
        else:
            raise ValueError(f"Key {key} has unsupported type {type(value)}")

    return msg


def from_ros_msg(msg: TensorDict):
    # We need a recursive defaultdict to handle nested tensors
    def recursive_default_dict():
        return defaultdict(recursive_default_dict)

    target = recursive_default_dict()

    for tensor_msg in msg.tensors:
        target_keys = tensor_msg.name.split("/")
        target_nested = target
        for key in target_keys[:-1]:
            target_nested = target_nested[key]

        msg_data = None
        try:
            if tensor_msg.uint8_data:
                msg_data = tensor_msg.uint8_data
            elif tensor_msg.float32_data:
                msg_data = tensor_msg.float32_data
            elif tensor_msg.float64_data:
                msg_data = tensor_msg.float64_data
            elif tensor_msg.int32_data:
                msg_data = tensor_msg.int32_data
            elif tensor_msg.int64_data:
                msg_data = tensor_msg.int64_data
            elif tensor_msg.bool_data:
                msg_data = tensor_msg.bool_data
            else:
                raise ValueError(f"No data in tensor {tensor_msg.name}")

            target_nested[target_keys[-1]] = np.array(list(msg_data)).reshape(
                tuple(tensor_msg.shape)
            )
        except ValueError as e:
            import rospy

            rospy.logerr(
                f"Failed to convert tensor {tensor_msg.name} with data length {len(msg_data) if msg_data else None} and shape {tensor_msg.shape}: {e}"
            )

    # Convert the nested defaultdict to a nested dict
    def convert_to_dict(target):
        if isinstance(target, defaultdict):
            return {key: convert_to_dict(value) for key, value in target.items()}
        else:
            return target

    return convert_to_dict(target)


if __name__ == "__main__":
    tensor = {
        "a": np.array([1, 2, 3]),
        "b": np.array([4, 5, 6]),
        "c": {
            "d": np.array([7, 8, 9]),
            "e": {
                "f": np.array([10, 11, 12]),
                "g": np.array([13, 14, 15]),
            },
            "h": np.array([16, 17, 18]),
        },
        "i": np.array([19, 20, 21]),
        "j": {
            "k": np.array([22, 23, 24]),
            "l": np.array([25, 26, 27]),
        },
    }
    print(from_ros_msg(to_ros_msg(tensor)))
