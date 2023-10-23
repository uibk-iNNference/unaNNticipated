import numpy as np
from innfrastructure import models
from tensorflow.keras import layers


def calculate_multiplications_until(target_model: str, target_layer: str):
    m = models.get_model(target_model)

    total_multplications = 0
    for layer in m.layers:
        if "conv" not in layer.name:
            continue

        output_shape = layer.output_shape[1:]
        kernel_size = layer.kernel_size
        filters = layer.filters

        multiplications = np.prod(output_shape) * np.prod(kernel_size) * filters
        total_multplications += multiplications

        if layer.name == target_layer:
            break

    return total_multplications


def count_parameters_until(target_model: str, target_layer: str):
    m = models.get_model(target_model)

    kernel_parameters = 0
    for layer in m.layers:
        if "conv" not in layer.name:
            continue

        kernels = layer.get_weights()[0]
        kernel_parameters += np.prod(kernels.shape)

        print(f"{layer.name}:\t{get_layer_configuration(layer)}")

        if layer.name == target_layer:
            print(f"\nTarget {layer.name} has {np.prod(kernels.shape)} parameters")
            break

    return kernel_parameters


def get_layer_configuration(layer) -> str:
    if not isinstance(layer, layers.Conv2D):
        return None

    input_shape = layer.input_shape
    k = layer.kernel_size[0]
    f = layer.filters

    return f"f{f}_k{k}_i{'x'.join([str(v) for v in input_shape[1:]])}"


def main():
    target_model = "cifar10_medium"

    target_layer = "conv2d_11"
    total_parameters = count_parameters_until(target_model, target_layer)
    print(
        f"Model {target_model} has {total_parameters} up to and including {target_layer}"
    )

    return


if __name__ == "__main__":
    main()
