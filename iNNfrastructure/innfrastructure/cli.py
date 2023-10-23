from werkzeug.serving import run_simple
import click
import os
import json
import yaml

import numpy as np


@click.group()
def cli():
    pass


# Prediction
@cli.command()
@click.argument("model_type")
@click.argument("input_path", type=click.Path(exists=True, readable=True))
@click.option(
    "-b",
    "--batch_dissemination",
    is_flag=True,
    help="Whether to predict batches sample by sample (default: False)",
)
@click.option(
    "-l",
    "--log_device_placement",
    is_flag=True,
    help="Whether to log device placement (default: False)",
)
@click.option(
    "--cublas_workspace_config",
    type=str,
    default=None,
    help="Value for the CUBLAS_WORKSPACE_CONFIG variable",
)
@click.option(
    "-d",
    "--device",
    type=str,
    default="/device:CPU:0",
    help="Device to predict on (default: /device:CPU:0)",
)
@click.option("--output_path", type=click.Path(), default=None)
@click.option(
    "--config_path", type=click.Path(exists=True, readable=True), default=None
)
@click.option(
    "-p", "--profiler_logdir", default=None, help="Outoput dir for profiler logs."
)
def predict(
    model_type: str,
    input_path: str,
    batch_dissemination: bool,
    log_device_placement: bool,
    cublas_workspace_config: str,
    device: str,
    output_path: str,
    config_path: str,
    profiler_logdir: str,
):
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # for safety

    if cublas_workspace_config is not None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_workspace_config

    import tensorflow as tf

    tf.debugging.set_log_device_placement(log_device_placement)

    from . import predictions, metadata

    if device not in metadata.get_devices():
        raise ValueError(f"Device {device} not found. :(")
    with tf.device(device):
        # assume that more than one sample is stored
        samples = np.load(input_path)

        if profiler_logdir:
            tf.profiler.experimental.start(
                logdir=profiler_logdir,
                options=tf.profiler.experimental.ProfilerOptions(python_tracer_level=1),
            )

        output, actual_device = predictions.predict(
            model_type, samples, batch_dissemination
        )
        assert (
            device == actual_device
        )  # sanity check, because this WILL break our experiments if it happens

        if profiler_logdir:
            tf.profiler.experimental.stop()

        if type(output) is list:
            converted_results = [
                metadata.convert_np_to_json(prediction) for prediction in output
            ]
        else:
            converted_results = metadata.convert_np_to_json(output)

    full_dict = dict()
    full_dict["cpu_info"] = metadata.get_cpu_info()
    full_dict["memory"] = metadata.get_memory()
    full_dict["hostname"] = metadata.get_hostname()
    if device is not None:
        full_dict["device"] = metadata.get_device_info(actual_device)
    full_dict["prediction"] = converted_results
    full_dict["model_type"] = model_type
    full_dict["batch_dissemination"] = batch_dissemination
    # full_dict["executing_commit"] = metadata.get_commit()

    if output_path is None:
        click.echo(json.dumps(full_dict, indent=2))
        return

    with open(output_path, "w") as output_file:
        json.dump(full_dict, output_file)


# 2. GCloud
# TODO: add help for menu items
@cli.group()
def gcloud():
    pass


@gcloud.command(
    name="start",
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    ),
)
@click.pass_context
@click.argument("name")
@click.argument("zone")
def gcloud_start(ctx, name: str, zone: str):
    from . import gcloud

    additional_args = dict([item.strip("--").split("=") for item in ctx.args])

    instance_config = gcloud.GcloudInstanceConfig(name, zone)
    instance_config.add_arguments(additional_args)
    click.echo(f"Starting instance {name} in {zone}")
    click.echo(f"Additional arguments:\n{additional_args}")
    gcloud.create_instance(instance_config)
    click.echo("Done")


@gcloud.command(name="list")
def gcloud_list():
    from . import gcloud

    instance_list = gcloud.get_instances()
    click.echo(instance_list)


@gcloud.command(name="stop")
@click.argument("name")
@click.argument("zone")
def gcloud_stop(name: str, zone: str):
    from . import gcloud

    instance_config = gcloud.GcloudInstanceConfig(name, zone)

    click.echo(f"Stopping instance {name} in {zone}")
    gcloud.stop_instance(instance_config)
    click.echo("Done")


@gcloud.command(name="delete")
@click.argument("name")
@click.argument("zone")
def gcloud_delete(name: str, zone: str):
    from . import gcloud

    instance_config = gcloud.GcloudInstanceConfig(name, zone)

    click.echo(f"Deleting instance {name} in {zone}")
    gcloud.delete_instance(instance_config)
    click.echo("Done")

# 4. Comparison
@cli.command()
@click.argument("layout_path", type=click.Path(exists=True, readable=True))
@click.option("--output_path", type=click.Path())
def compare(layout_path: str, output_path: str = None):
    from . import compare, metadata

    with open(layout_path, "r") as f:
        layout = yaml.safe_load(f)

    # replace paths with loaded predictions
    def load_path(path):
        """load result from path depending on file extension"""
        if path.endswith("json"):
            with open(path, "r") as f:
                values = yaml.safe_load(f)
                return metadata.convert_json_to_np(values["prediction"])

        if path.endswith("npy"):
            return np.load(path)

        raise ValueError(f"Unknown file extension in path {path}")

    for results in layout.values():
        # iterate over experiments
        for name, path in results.items():
            results[name] = load_path(path)

    comparison = compare.generate_comparison(layout)

    if output_path is None:
        click.echo(comparison.to_markdown())
        return

    comparison.to_csv(f"comparison_{output_path}.csv")


if __name__ == "__main__":
    cli()
