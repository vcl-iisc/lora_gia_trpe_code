"""
Control script to launch some attack that attempts to breach privacy federated learning in a given case
with the overall goal of recovering user data, e.g. image data, directly as much as possible.
"""

# Set SDPA kernel preferences BEFORE importing torch modules
import torch
# torch.backends.cuda.enable_flash_sdp(False)
# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_math_sdp(True)

# Also set CUDA backend preferences to ensure math kernel is used
# torch.set_float32_matmul_precision('highest')

import hydra
from omegaconf import OmegaConf

import datetime
import time
import logging

import breaching

import os

os.environ["HYDRA_FULL_ERROR"] = "0"
log = logging.getLogger(__name__)


def _unpack_reconstruct_output(result):
    """Support attacks that return either 2 or 3 values."""
    if not isinstance(result, tuple):
        raise TypeError(f"Unexpected reconstruct output type: {type(result)!r}")
    if len(result) == 2:
        reconstructed_user_data, stats = result
        all_reconstructions = None
    elif len(result) == 3:
        reconstructed_user_data, stats, all_reconstructions = result
    else:
        raise ValueError(f"Unexpected reconstruct output length: {len(result)}")
    return reconstructed_user_data, stats, all_reconstructions


def main_process(process_idx, local_group_size, cfg):
    """This function controls the central routine."""
    local_time = time.time()
    setup = breaching.utils.system_startup(process_idx, local_group_size, cfg)

    # Propose a model architecture:
    # (Replace this line with your own model if you want)
    model, loss_fn = breaching.cases.construct_model(
        cfg.case.model,
        cfg.case.data,
        cfg.case.server.pretrained,
        cfg_case=cfg.case,
    )
    log.info(f"Number of Trainable Parameters: {breaching.utils.count_trainable_parameters(model)}")
    # Instantiate server and vet model
    # This is a no-op for an honest-but-curious server, but a malicious-model server can modify the model in this step.
    server = breaching.cases.construct_server(model, loss_fn, cfg.case, setup)
    model = server.vet_model(model)

    # Instantiate user and attacker
    user = breaching.cases.construct_user(model, loss_fn, cfg.case, setup)
    attacker = breaching.attacks.prepare_attack(model, loss_fn, cfg.attack, setup)
    # Summarize startup:
    breaching.utils.overview(server, user, attacker)

    # Simulate a simple FL protocol
    shared_user_data, payloads, true_user_data = server.run_protocol(user)

    # Run an attack using only payload information and shared data
    reconstruct_output = attacker.reconstruct(payloads, shared_user_data, server.secrets, dryrun=cfg.dryrun)
    reconstructed_user_data, stats, _ = _unpack_reconstruct_output(reconstruct_output)

    # How good is the reconstruction?
    metrics = breaching.analysis.report(
        reconstructed_user_data, true_user_data, payloads, model, cfg_case=cfg.case, setup=setup
    )

    # Save to summary:
    breaching.utils.save_summary(cfg, metrics, stats, user.counted_queries, time.time() - local_time)
    # Save to output folder:
    breaching.utils.dump_metrics(cfg, metrics)
    if cfg.save_reconstruction:
        breaching.utils.save_reconstruction(reconstructed_user_data, payloads, true_user_data, cfg)


@hydra.main(config_path="breaching/config", config_name="cfg", version_base="1.1")
def main_launcher(cfg):
    """This is boiler-plate code for the launcher."""

    log.info("--------------------------------------------------------------")
    log.info("-----Launching federating learning breach experiment! --------")

    launch_time = time.time()
    if cfg.seed is None:
        cfg.seed = torch.randint(0, 2**32 - 1, (1,)).item()

    log.info(OmegaConf.to_yaml(cfg))
    breaching.utils.initialize_multiprocess_log(cfg)  # manually save log configuration
    main_process(0, 1, cfg)

    log.info("-------------------------------------------------------------")
    log.info(
        f"Finished computations {cfg.name} with total train time: "
        f"{str(datetime.timedelta(seconds=time.time() - launch_time))}"
    )
    log.info("-----------------Job finished.-------------------------------")


if __name__ == "__main__":
    main_launcher()
