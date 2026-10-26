"""This script computes a suite of benchmark numbers for the given attack.


The arguments from the default config carry over here.
"""

import hydra
from omegaconf import OmegaConf, open_dict

import datetime
import time
import logging
import breaching

import os

os.environ["HYDRA_FULL_ERROR"] = "1"  # Enable full Hydra errors for debugging
log = logging.getLogger(__name__)


def main_process(process_idx, local_group_size, cfg, num_trials=100, job_name=None):
    """This function controls the central routine."""
    total_time = time.time()  # Rough time measurements here
    setup = breaching.utils.system_startup(process_idx, local_group_size, cfg)

    # To modify the cfg.case.model to include the posembed_trainable argument from cfg.attack, we need to open the cfg for editing.
    # To bypass Hydra's immutability, we can use the open_dict context manager. 
    # This allows us to modify the cfg in-place without creating a new copy.
    if hasattr(cfg.attack.objective, "posembed_trainable"):
        if cfg.attack.objective.posembed_trainable:
            with open_dict(cfg):
                if isinstance(cfg.case.model, str):
                    # Passing the posembed_trainable argument from attack config to case model config for backward compatibility. 
                    # This is needed for attacks that want to optimize over positional embeddings, 
                    # but the case model config only accepts a boolean for posembed_trainable and not the scale.
                    # Also, in the cfg.case only the model_name is defined, so we need to convert it to a dict for the model constructor.
                    cfg.case.model = {"name": cfg.case.model, 
                                    "posembed_trainable": cfg.attack.objective.posembed_trainable}
                elif isinstance(cfg.case.model, dict) and "posembed_trainable" not in cfg.case.model:
                    cfg.case.model["posembed_trainable"] = cfg.attack.objective.posembed_trainable
            if cfg.attack.objective.apply_block_pos_embed:
                log.info("Applying BLOCK positional embedding to every layer")
                with open_dict(cfg):
                    cfg.case.model["apply_block_pos_embed"] = True
        else:
            if "posembed" in cfg.attack.objective.type.lower():
                log.warning("""The argument attack.objective.posembed_trainable is set to False,
                            but the objective type contains 'posembed'. This might be a configuration error.""")
                raise ValueError("""The argument attack.objective.posembed_trainable is set to False, but the objective type contains 'posembed'. 
                                    If attack.objective.posembed_trainable=False is the desired behavior, consider using a different objective type. 
                                    For example:
                                    cosine-grad-cossim-posembed -> cosine-similarity
                                    euclidean-grad-cossim-posembed -> euclidean
                                    """)
    else:
        if "posembed" in cfg.attack.objective.type.lower():
            log.warning("""The argument attack.objective.posembed_trainable is not set,
                        but the objective type contains 'posembed'. This might be a configuration error.""")
            raise ValueError("""The argument attack.objective.posembed_trainable is not set, but the objective type contains 'posembed'. 
                                If attack.objective.posembed_trainable=False is the desired behavior, consider using a different objective type. 
                                For example:
                                cosine-grad-cossim-posembed -> cosine-similarity
                                euclidean-grad-cossim-posembed -> euclidean
                                """)

    # MODEL IS CONSTRUCTED HERE; the attack and user construction happens later, 
    # after we have selected the user and know which data they have, so that we can pass the data shape to the attack if needed.
    # MODEL CONFIG from case config is passed here, so any custom information like embed_layer, depth, etc. is pased here
    model, loss_fn = breaching.cases.construct_model(
        cfg.case.model,
        cfg.case.data,
        cfg.case.server.pretrained, # This is in server configs which are loaded by default, but now added in case config files as well
        cfg.case.server.pretrained_path,
        cfg_case=cfg.case,
    )
    log.info(f"Model: {cfg.case.model}")
    log.info(f"Number of Trainable Parameters: {breaching.utils.count_trainable_parameters(model)}")

    if cfg.num_trials is not None:
        num_trials = cfg.num_trials

    server = breaching.cases.construct_server(model, loss_fn, cfg.case, setup)
    model = server.vet_model(model)
    # log.warning(f"PEFT Configuration: {cfg.case.peft}")
    attacker = breaching.attacks.prepare_attack(model, loss_fn, cfg.attack, setup, cfg_peft=cfg.case.peft) 
    # Pass the peft type to the attack for it to decide whether to optimize over positional embeddings or not. 

    user_idx = cfg.case.user.get("user_idx", None)
    if user_idx is not None:
        log.info("The argument user_idx is disregarded during the benchmark. Data selection is fixed.")
    log.info(
        f"Partitioning is set to {cfg.case.data.partition}. Make sure there exist {num_trials} users in this scheme."
    )

    with open_dict(cfg):
        cfg.case.user.user_idx = -1
    run = 0
    requested_num_users = int(cfg.case.get("num_users", 10))  # Default to num_trials if not specified in config
    if requested_num_users < 1:
        raise ValueError(f"num_users must be >= 1, got {requested_num_users}.")
    num_users = requested_num_users
    log.info(f"Requested num_users per trial: {requested_num_users}.")

    if cfg.case.data.name in ("CIFAR10", "CIFAR100"):
        full_split_loader = breaching.cases.construct_dataloader(
            cfg.case.data, cfg.case.impl, user_idx=None, return_full_dataset=True
        )
        total_split_samples = len(full_split_loader.dataset)
        samples_per_user = int(cfg.case.user.num_data_points)
        log.info(f"Total samples in {cfg.case.data.examples_from_split} split of {cfg.case.data.name}: {total_split_samples}.")
        log.info(f"Samples per user: {samples_per_user}.")
        if samples_per_user < 1:
            raise ValueError(f"case.user.num_data_points must be >= 1, got {samples_per_user}.")
        max_users_by_split = total_split_samples // samples_per_user
        if max_users_by_split < 1:
            raise ValueError(
                f"Split {cfg.case.data.examples_from_split} of {cfg.case.data.name} has {total_split_samples} samples, "
                f"insufficient for num_data_points={samples_per_user}."
            )
        num_users = min(requested_num_users, max_users_by_split)
        log.info(f"Final num_users per trial: {num_users}.")
        if num_users < requested_num_users:
            log.warning(
                "Capping num_users from %d to %d for %s %s split: total_samples=%d, samples_per_user=%d.",
                requested_num_users,
                num_users,
                cfg.case.data.name,
                cfg.case.data.examples_from_split,
                total_split_samples,
                samples_per_user,
            )
    user = None
    overall_metrics = []
    stats = {}
    log.info(f"Starting benchmark with num_trials={num_trials}.")
    while run < num_trials:
        log.info(f"Starting trial {run} with num_users={num_users}.")
        local_time = time.time()
        # Select data that has not been seen before:
        if num_trials > 0 and run > 0 and cfg.case.user.user_idx >= 0:
            # Increment the user index to select a new user for the next trial.
            # This assumes that the user indices are sequential and that the data partitioning allows for this.
            cfg.case.user.user_idx = -1  # Reset to -1 to ensure we start from the beginning of the user index range for each trial.
        cfg.case.user.user_idx += 1
        round_start_user_idx = cfg.case.user.user_idx
        try:
            if num_users > 1:
                # Build a distinct user object per participant in this FL round.
                users = []
                for offset in range(num_users):
                    cfg.case.user.user_idx = round_start_user_idx + offset
                    users.append(breaching.cases.construct_user(model, loss_fn, cfg.case, setup))
                    log.info(f"User type: {users[-1].__class__.__name__}")
                user = users
                # Keep global pointer at the last consumed user id for next trial.
                cfg.case.user.user_idx = round_start_user_idx + num_users - 1
            else:
                user = breaching.cases.construct_user(model, loss_fn, cfg.case, setup)
                log.info(f"User type: {user.__class__.__name__}")
            
            if isinstance(user, list) and len(user) > 0:
                log.info(f"User Batch Size: {user[0].dataloader.batch_size}")
            else:
                log.info(f"User Batch Size: {user.dataloader.batch_size}") # OK
        except Exception as e:
            log.info("Cannot find other valid users. Finishing benchmark.")
            log.info(e)
            break
        current_user = user[0] if isinstance(user, list) else user
        if cfg.case.data.modality == "text":
            dshape = current_user.dataloader.dataset[0]["input_ids"].shape
            data_shape_mismatch = any([d != d_ref for d, d_ref in zip(dshape, cfg.case.data.shape)])
        else:
            data_shape_mismatch = False  # Handled by preprocessing for images
        if len(current_user.dataloader.dataset) < current_user.num_data_points or data_shape_mismatch:
            log.info(f"Skipping user {current_user.user_idx} (has not enough data or data shape mismatch).")
        else:
            users_to_attack = user if isinstance(user, list) else [user]
            log.info(
                "Now evaluating %d user(s) in trial %d: %s.",
                len(users_to_attack),
                run,
                [u.user_idx for u in users_to_attack],
            )

            for single_user in users_to_attack:
                try:
                    shared_user_data, payloads, true_user_data = server.run_protocol(single_user)
                    log.info(f"FL protocol executed successfully for trial {run}. Shared data chunks emitted: {len(shared_user_data)}.")
                except Exception as e:
                    log.info("FL protocol execution for trial %d broke down with error %s.", run, str(e))
                    run += 1
                    continue
                try:
                    uid = int(single_user.user_idx)
                    # WORKS FOR HONEST_AND_CURIOUS_WITH_MEMORY: 
                    # We look for emitted shared-data chunks that have matching user_idx in their metadata, and align the payloads to those chunks.
                    # DOES NOT WORK FOR HONEST_AND_CURIOUS:
                    # NO METADATA IN SHARED-DATA CHUNKS, so we cannot align the payloads to the user. 
                    # We proceed with all emitted shared-data chunks for reconstruction, but log a warning that the alignment might be incorrect.
                    per_user_indices = [
                        idx
                        for idx, item in enumerate(shared_user_data)
                        if int(item.get("metadata", {}).get("user_idx", -1)) == uid
                    ]
                    # NEEDED FOR HONEST AND CURIOUS ONLY SETTING, OTHERWISE WE WOULD HAVE ALIGNED TO THE WRONG USER'S DATA (IF MULTIPLE USERS):
                    if len(per_user_indices) == 0:
                        log.info("No emitted shared-data chunk with matching user_idx in metadata for trial %d user %s.", run, str(single_user.user_idx))
                        log.info("May not contain user_idx in metadata in the shared data chunks")
                        if len(shared_user_data) > 0:
                            log.info("Emitted shared-data chunks metadata: %s", [item.get("metadata", {}) for item in shared_user_data])
                            per_user_indices = list(range(len(shared_user_data)))
                            log.info("Proceeding with all emitted shared-data chunks for trial %d user %s.", run, str(single_user.user_idx))
                        else:
                            log.info("No emitted shared-data chunks at all for trial %d. Skipping reconstruction for user %s.", run, str(single_user.user_idx))
                            continue
                    log.info(
                        "Trial %d user %d: found %d emitted shared-data chunk(s) with matching user_idx in metadata.",
                        run,
                        uid,
                        len(per_user_indices)
                    )
                    per_user_shared_data = [shared_user_data[idx] for idx in per_user_indices]
                    if len(per_user_shared_data) == 0:
                        log.info("No per-user shared data emitted for trial %d user %s.", run, str(single_user.user_idx))
                        continue
                    # Align payloads exactly to this user's emitted shared-data entries.
                    per_user_payloads = [payloads[idx] for idx in per_user_indices]
                    log.info(
                        "Trial %d user %d: reconstructing with %d emitted chunk(s), payload indices=%s.",
                        run,
                        uid,
                        len(per_user_payloads),
                        per_user_indices,
                    )
                    for local_chunk_id, global_payload_idx in enumerate(per_user_indices):
                        metadata = per_user_shared_data[local_chunk_id].get("metadata", {})
                        log.info(
                            "Trial %d user %d: chunk %d/%d uses payload[%d], metadata=%s.",
                            run,
                            uid,
                            local_chunk_id + 1,
                            len(per_user_indices),
                            global_payload_idx,
                            metadata,
                        )
                    if isinstance(true_user_data, list):
                        matching_true_user_data = true_user_data[users_to_attack.index(single_user)]
                    else:
                        matching_true_user_data = true_user_data
                    optimal_reconstruction, stats, all_reconstruction = attacker.reconstruct(per_user_payloads, 
                                                                                            per_user_shared_data, 
                                                                                            server.secrets, dryrun=cfg.dryrun)

                    # log.info(f"stats from reconstruction: {stats}")
                except Exception as e:
                    log.info(
                        "Reconstruction for trial %d user %s broke down with error %s.",
                        run,
                        str(single_user.user_idx),
                        str(e),
                    )
                    continue
                try:
                    if all_reconstruction is None and optimal_reconstruction is not None:
                        reconstruction = optimal_reconstruction
                        metrics = breaching.analysis.report(
                            reconstruction,
                            matching_true_user_data,
                            per_user_payloads,
                            server.model,
                            order_batch=True,
                            compute_full_iip=True,
                            compute_rpsnr=True,
                            compute_ssim=True,
                            cfg_case=cfg.case,
                            setup=setup,
                        )
                        metrics["queries"] = single_user.counted_queries
                        metrics["user_idx"] = int(single_user.user_idx)

                        per_user_job_name = (
                            f"{job_name}_user{single_user.user_idx}" if job_name is not None else f"user{single_user.user_idx}"
                        )
                        breaching.utils.save_summary(
                            cfg,
                            metrics,
                            stats,
                            time.time() - local_time,
                            original_cwd=False,
                            job_name=per_user_job_name,
                        )
                        overall_metrics.append(metrics)
                    elif all_reconstruction is not None:
                        for idx, reconstruction in enumerate(all_reconstruction):
                            metrics = breaching.analysis.report(
                                reconstruction,
                                matching_true_user_data,
                                per_user_payloads,
                                server.model,
                                order_batch=True,
                                compute_full_iip=True,
                                compute_rpsnr=True,
                                compute_ssim=True,
                                cfg_case=cfg.case,
                                setup=setup,
                            )
                            metrics["queries"] = single_user.counted_queries
                            metrics["user_idx"] = int(single_user.user_idx)
                            metrics["reconstruction_id"] = idx

                            per_user_job_name = (
                                f"{job_name}_user{single_user.user_idx}_recon{idx}" if job_name is not None else f"user{single_user.user_idx}_recon{idx}"
                            )
                            breaching.utils.save_summary(
                                cfg,
                                metrics,
                                stats,
                                time.time() - local_time,
                                original_cwd=False,
                                job_name=per_user_job_name,
                                idx=idx,
                            )
                            overall_metrics.append(metrics)
                        log.info(f"Overall Metrics: {len(overall_metrics)}")
                    else:
                        log.info(
                            "No valid reconstruction available for trial %d user %s. Skipping metrics computation.",
                            run,
                            str(single_user.user_idx),
                        )
                        continue
                    
                except Exception as e:
                    log.info(
                        "Metrics computation for trial %d user %s broke down with error %s.",
                        run,
                        str(single_user.user_idx),
                        str(e),
                    )
                    continue
                try:
                    if cfg.save_reconstruction:
                        with open_dict(cfg):
                            cfg.case.user.user_idx = int(single_user.user_idx)
                        breaching.utils.save_reconstruction(optimal_reconstruction, per_user_payloads, matching_true_user_data, cfg)
                    if cfg.dryrun:
                        break
                except Exception as e:  # noqa
                    log.info(
                        "Saving reconstruction for trial %d user %s broke down with error %s.",
                        run,
                        str(single_user.user_idx),
                        str(e),
                    )
        run += 1

    # Compute average statistics:
    log.info(f"Computing average metrics across all {len(overall_metrics)} reconstructions.")
    average_metrics = breaching.utils.avg_n_dicts(overall_metrics)
    # Save global summary:
    breaching.utils.save_summary(
        cfg, average_metrics, stats, time.time() - total_time, original_cwd=False, table_name="BENCHMARK_breach", job_name=job_name
    )

    # log.info(f"Computing standard deviation metrics across all {len(overall_metrics)} reconstructions.")
    # std_metrics = breaching.utils.std_n_dicts(overall_metrics)
    # breaching.utils.save_summary(
    #     cfg, std_metrics, stats, time.time() - total_time, original_cwd=False, table_name="BENCHMARK_breach_std", job_name=job_name
    # )


@hydra.main(config_path="breaching/config", config_name="cfg", version_base="1.1")
def main_launcher(cfg):
    """This is boiler-plate code for the launcher."""

    log.info("--------------------------------------------------------------")
    log.info("-----Launching federating learning breach experiment! --------")

    launch_time = time.time()
    if cfg.seed is None:
        cfg.seed = int.from_bytes(os.urandom(4), "big")  # randomized per run

    # Extract job_name from config if provided
    job_name = cfg.get("job_name", None)
    if job_name:
        log.info(f"Running with job name: {job_name}")

    log.info(OmegaConf.to_yaml(cfg))
    breaching.utils.initialize_multiprocess_log(cfg)  # manually save log configuration
    main_process(0, 1, cfg, job_name=job_name)

    log.info("-------------------------------------------------------------")
    log.info(
        f"Finished computations {cfg.name} with total train time: "
        f"{str(datetime.timedelta(seconds=time.time() - launch_time))}"
    )
    log.info("-----------------Job finished.-------------------------------")


if __name__ == "__main__":
    main_launcher()
