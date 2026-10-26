"""System utilities."""

import socket
import sys

import os
import csv
import yaml

import torch
import torchvision

import random
import numpy as np
import datetime

import hydra
from omegaconf import OmegaConf, open_dict

import logging

log = logging.getLogger(__name__)

def system_startup(process_idx, local_group_size, cfg):
    """Decide and print GPU / CPU / hostname info. Generate local distributed setting if running in distr. mode."""
    log = get_log(cfg)
    torch.backends.cudnn.benchmark = cfg.case.impl.benchmark
    sharing_strategy = cfg.case.impl.sharing_strategy
    if sharing_strategy in torch.multiprocessing.get_all_sharing_strategies():
        torch.multiprocessing.set_sharing_strategy(sharing_strategy)
    else:
        log.warning(
            f"Multiprocessing sharing strategy {sharing_strategy} not supported on this platform. "
            f"Available strategies: {torch.multiprocessing.get_all_sharing_strategies()}. "
            f"Continuing with default."
        )
    huggingface_offline_mode(cfg.case.impl.enable_huggingface_offline_mode)
    # 100% reproducibility?
    if cfg.case.impl.deterministic:
        set_deterministic()
    if cfg.seed is not None:
        set_random_seed(cfg.seed + 10 * process_idx)

    dtype = getattr(torch, cfg.case.impl.dtype)  # :> dont mess this up

    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if cfg.case.impl.enable_gpu_acc:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{process_idx}")
        elif mps_available:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    setup = dict(device=device, dtype=dtype)
    python_version = sys.version.split(" (")[0]
    log.info(f"Platform: {sys.platform}, Python: {python_version}, PyTorch: {torch.__version__}")
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    log.info(f"CPUs: {torch.get_num_threads()}, GPUs: {gpu_count} on {socket.gethostname()}.")

    if device.type == "cuda":
        torch.cuda.set_device(process_idx)
        log.info(f"GPU : {torch.cuda.get_device_name(device=device)}")
    elif device.type == "mps":
        log.info("GPU : Apple MPS")

    # if not torch.cuda.is_available() and not cfg.dryrun:
    #     raise ValueError('No GPU allocated to this process. Running in CPU-mode is likely a bad idea. Complain to your admin.')

    return setup


def is_main_process():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def get_log(cfg, name=os.path.basename(__file__)):
    """Solution via https://github.com/facebookresearch/hydra/issues/1126#issuecomment-727826513"""
    if is_main_process():
        logging.config.dictConfig(OmegaConf.to_container(cfg.job_logging_cfg, resolve=True))
        logger = logging.getLogger(name)
    else:

        def logger(*args, **kwargs):
            pass

        logger.info = logger
    return logger


def initialize_multiprocess_log(cfg):
    with open_dict(cfg):
        # manually save log config to cfg
        log_config = hydra.core.hydra_config.HydraConfig.get().job_logging
        # but resolve any filenames
        cfg.job_logging_cfg = OmegaConf.to_container(log_config, resolve=True)
        cfg.original_cwd = hydra.utils.get_original_cwd()


def save_summary(cfg, metrics, stats, local_time, original_cwd=True, table_name="breach", job_name=None, idx=None):
    """Save two summary tables. A detailed table of iterations/loss+acc and a summary of the end results."""
    log.info(f"Saving summary for {cfg.case.name} user {cfg.case.user.user_idx} with metrics {metrics}") # and stats {stats}.")
    stats = stats or {}
    # 1) detailed table:
    train_loss = stats.get("train_loss", [])
    for step in range(len(train_loss)):
        iteration = dict()
        for key in stats:
            iteration[key] = stats[key][step] if step < len(stats[key]) else None
        save_to_table(".", f"{cfg.attack.type}_convergence_results", dryrun=cfg.dryrun, **iteration)

    try:
        local_folder = os.getcwd().split("outputs/")[1]
    except IndexError:
        local_folder = ""

    # 2) save a reduced summary
    summary = dict(
        name=cfg.name,
        usecase=cfg.case.name,
        model=cfg.case.model,
        datapoints=cfg.case.user.num_data_points,
        model_state=cfg.case.server.model_state,
        attack=cfg.attack.type,
        attacktype=cfg.attack.attack_type,
        **{k: v for k, v in metrics.items() if k != "order"},
        score=stats.get("opt_value"),
        total_time=str(datetime.timedelta(seconds=local_time)).replace(",", ""),
        user_type=cfg.case.user.user_type,
        gradient_noise=cfg.case.user.local_diff_privacy.gradient_noise,
        seed=cfg.seed,
        # dump extra values from here:
        **{f"ATK_{k}": v for k, v in cfg.attack.items()},
        **{k: v for k, v in cfg.case.items() if k not in ["name", "model"]},
        folder=local_folder,
    )

    # Use job_name in path if provided
    if job_name:
        location = os.path.join(cfg.original_cwd, "tables", job_name) if original_cwd else os.path.join("tables", job_name)
        if os.path.exists(location):
            count = len([f for f in os.listdir(os.path.dirname(location)) if f.startswith(job_name)])
            location = os.path.join(cfg.original_cwd, "tables", f"{job_name}_{count+1}") if original_cwd else os.path.join("tables", f"{job_name}_{count+1}")
    else:
        location = os.path.join(cfg.original_cwd, "tables") if original_cwd else "tables"
        if os.path.exists(location):
            count = len([f for f in os.listdir(os.path.dirname(location)) if f.startswith("tables")])
            location = os.path.join(cfg.original_cwd, f"tables_{count+1}") if original_cwd else f"tables_{count+1}"
    
    log.info(f"Saving summary to {location} with table name {table_name}") # and summary {summary}.")
    if idx is None:
        save_to_table(location, f"{table_name}_{cfg.case.name}_{cfg.case.data.name}_reports", dryrun=cfg.dryrun, **summary)
    else:
        save_to_table(location, f"{table_name}_{cfg.case.name}_{cfg.case.data.name}_reports_{idx}", dryrun=cfg.dryrun, **summary)


def save_to_table(out_dir, table_name, dryrun, **kwargs):
    """Save keys to .csv files. Function adapted from Micah."""
    # Check for file
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    fname = os.path.join(out_dir, f"table_{table_name}.csv")
    fieldnames = list(kwargs.keys())

    # Read or write header
    try:
        with open(fname, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)  # noqa  # this line is testing the header
            # assert header == fieldnames[:len(header)]  # new columns are ok, but old columns need to be consistent
            # dont test, always write when in doubt to prevent erroneous table rewrites
    except Exception as e:  # noqa
        if not dryrun:
            # print('Creating a new .csv table...')
            with open(fname, "w") as f:
                writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
                writer.writeheader()
        else:
            pass
            # print(f'Would create new .csv table {fname}.')

    # Write a new row
    if not dryrun:
        # Add row for this experiment
        with open(fname, "a") as f:
            writer = csv.DictWriter(f, delimiter="\t", fieldnames=fieldnames)
            writer.writerow(kwargs)
        # print('\nResults saved to ' + fname + '.')
    else:
        pass
        # print(f'Would save results to {fname}.')


def set_random_seed(seed=233):
    """."""
    torch.manual_seed(seed + 1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + 2)
        torch.cuda.manual_seed_all(seed + 3)
    np.random.seed(seed + 4)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + 5)
    random.seed(seed + 6)
    # Can't be too careful :>


def set_deterministic():
    """Switch pytorch into a deterministic computation mode."""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import math

def std_n_dicts(dicts):
    if not dicts:
        return {}
    stds = {}
    # print(dicts[0])
    for key in dicts[0]:
        if key not in ['order']:
            try:
                value = dicts[0][key]
                if isinstance(value, list):
                    stds[key] = torch.tensor([[d[key][i] for d in dicts] for i in range(len(value))], dtype=torch.float32).std(dim=1, correction=1).tolist()
                else:
                    stds[key] = torch.tensor([d[key] for d in dicts], dtype=torch.float32).std(correction=1).item()
            except:
                print(key)
    return stds

def avg_n_dicts(dicts):
    """https://github.com/wronnyhuang/metapoison/blob/master/utils.py."""
    # given a list of dicts with the same exact schema, return a single dict with same schema whose values are the
    # key-wise average over all input dicts
    means = {}
    for dic in dicts:
        for key in dic:
            if key not in means:
                if isinstance(dic[key], list):
                    means[key] = [0 for entry in dic[key]]
                else:
                    means[key] = 0
            if isinstance(dic[key], list):
                for idx, entry in enumerate(dic[key]):
                    means[key][idx] += entry / len(dicts)
            else:
                means[key] += dic[key] / len(dicts)
    return means


def get_base_cwd():
    try:
        return hydra.utils.get_original_cwd()
    except ValueError:  # Hydra not initialized:
        return os.getcwd()


def overview(server, user, attacker):
    num_params, num_buffers = (
        sum([p.numel() for p in user.model.parameters()]),
        sum([b.numel() for b in user.model.buffers()]),
    )
    target_information = user.num_data_points * torch.as_tensor(server.cfg_data.shape).prod()
    print(f"Model architecture {user.model.name} loaded with {num_params:,} parameters and {num_buffers:,} buffers.")
    print(
        f"Overall this is a data ratio of {server.num_queries * num_params / target_information:7.0f}:1 "
        f"for target shape {[user.num_data_points, *server.cfg_data.shape]} given that num_queries={server.num_queries}."
    )
    print(user)
    print(server)
    print(attacker)


def count_trainable_parameters(model):
    """Return the number of trainable parameters in a model."""
    if model is None:
        return 0
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_reconstruction(
    reconstructed_user_data, server_payload, true_user_data, cfg, side_by_side=True, target_indx=None
):
    """If target_indx is not None, only the datapoints at target_indx will be saved to file."""
    os.makedirs("reconstructions", exist_ok=True)
    metadata = server_payload[0]["metadata"]
    if metadata["modality"] == "text":
        from breaching.cases.data.datasets_text import _get_tokenizer

        tokenizer = _get_tokenizer(
            server_payload[0]["metadata"]["tokenizer"],
            server_payload[0]["metadata"]["vocab_size"],
            cache_dir=cfg.case.data.path,
        )
        text_rec = tokenizer.batch_decode(reconstructed_user_data["data"])
        text_ref = tokenizer.batch_decode(true_user_data["data"])
        if target_indx is not None:
            text_rec = text_rec[target_indx]
            text_ref = text_ref[target_indx]

        filepath = os.path.join(
            "reconstructions", f"text_rec_{cfg.case.data.name}_{cfg.case.model}_user{cfg.case.user.user_idx}.txt"
        )

        with open(filepath, "w") as f:
            f.writelines(text_rec)
            if side_by_side:
                f.write("\n")
                f.write("========== GROUND TRUTH TEXT ===========")
                f.write("\n")
                f.writelines(text_ref)
    else:
        if hasattr(metadata, "mean"):
            dm = torch.as_tensor(metadata.mean)[None, :, None, None]
            ds = torch.as_tensor(metadata.std)[None, :, None, None]
        else:
            dm, ds = torch.tensor(
                0,
            ), torch.tensor(1)

        rec_denormalized = torch.clamp(reconstructed_user_data["data"].cpu() * ds + dm, 0, 1)
        ground_truth_denormalized = torch.clamp(true_user_data["data"].cpu() * ds + dm, 0, 1)
        if target_indx is not None:
            rec_denormalized = rec_denormalized[target_indx]
            ground_truth_denormalized = ground_truth_denormalized[target_indx]

        filepath = os.path.join(
            "reconstructions",
            f"img_rec_{cfg.case.data.name}_{cfg.case.model['name']}_{cfg.case.model['patch_size']}_user{cfg.case.user.user_idx}.png",
        )
        if os.path.exists(filepath):
            count = len([f for f in os.listdir("reconstructions") if f.startswith(f"img_rec_{cfg.case.data.name}_{cfg.case.model}_user{cfg.case.user.user_idx}")])
            filepath = os.path.join(
                "reconstructions",
                f"img_rec_{cfg.case.data.name}_{cfg.case.model['name']}_{cfg.case.model['patch_size']}_user{cfg.case.user.user_idx}_{count+1}.png",
            )

        labels = None
        if isinstance(true_user_data, dict) and "labels" in true_user_data:
            labels = true_user_data["labels"]
        elif isinstance(reconstructed_user_data, dict) and "labels" in reconstructed_user_data:
            labels = reconstructed_user_data["labels"]
        if labels is not None:
            if torch.is_tensor(labels):
                labels = labels.detach().cpu().tolist()
            elif not isinstance(labels, list):
                labels = [labels]
            if target_indx is not None:
                if torch.is_tensor(target_indx):
                    target_indx = target_indx.detach().cpu().tolist()
                if isinstance(target_indx, (list, tuple)):
                    labels = [labels[i] for i in target_indx]
                else:
                    labels = [labels[target_indx]]

        # Save individual images (reconstruction + ground truth) into separate subfolders,
        # in addition to the side-by-side grid written below.
        _save_individual_images(
            rec_denormalized,
            ground_truth_denormalized,
            labels,
            cfg,
            has_gt=side_by_side,
        )

        if not side_by_side:
            _save_titled_image_grid(
                rec_denormalized,
                None,
                labels,
                filepath,
                top_title="Rec", #onstructed",
            )
        else:
            _save_titled_image_grid(
                rec_denormalized,
                ground_truth_denormalized,
                labels,
                filepath,
                top_title="Rec",
                bottom_title="GT",
            )


def _save_titled_image_grid(rec_images, gt_images, labels, filepath, top_title, bottom_title=None):
    """Save a grid image with per-image top labels and side-by-side GT/Reconstruction."""
    from PIL import Image, ImageDraw, ImageFont

    padding = 2
    rec_label = top_title or "Rec"
    gt_label = bottom_title or "GT"

    rec_pil = [torchvision.transforms.functional.to_pil_image(img) for img in rec_images]
    gt_pil = None
    if gt_images is not None:
        gt_pil = [torchvision.transforms.functional.to_pil_image(img) for img in gt_images]

    img_w, img_h = rec_pil[0].size
    n = len(rec_pil)
    cols_per_item = 2 if gt_pil is not None else 1
    total_cols = n * cols_per_item

    grid_w = total_cols * img_w + (total_cols + 1) * padding
    grid_h = img_h + 2 * padding
    top_margin = 24 if labels else 8
    bottom_margin = 24
    canvas = Image.new("RGB", (grid_w, grid_h + top_margin + bottom_margin), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    # Use a TrueType font to avoid pixelated bitmap rendering.
    font_size = max(10, img_h // 20)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    def _text_width(text):
        if hasattr(draw, "textlength"):
            return draw.textlength(text, font=font)
        return len(text) * 6

    # Paste images
    y0 = top_margin + padding
    for idx in range(n):
        base_col = idx * cols_per_item
        x_rec = padding + base_col * (img_w + padding)
        canvas.paste(rec_pil[idx], (x_rec, y0))
        if gt_pil is not None:
            x_gt = padding + (base_col + 1) * (img_w + padding)
            canvas.paste(gt_pil[idx], (x_gt, y0))

    # Top labels (per image/pair)
    if labels:
        for idx, label in enumerate(labels):
            label_text = f"Label: {label}"
            pair_x = padding + (idx * cols_per_item) * (img_w + padding)
            pair_w = cols_per_item * img_w + (cols_per_item - 1) * padding
            label_x = pair_x + (pair_w - _text_width(label_text)) / 2
            draw.text((label_x, 2), label_text, fill=(0, 0, 0), font=font)

    # Bottom labels (under each image)
    label_y = top_margin + grid_h + 2
    for idx in range(n):
        base_col = idx * cols_per_item
        x_rec = padding + base_col * (img_w + padding)
        rec_x = x_rec + (img_w - _text_width(rec_label)) / 2
        draw.text((rec_x, label_y), rec_label, fill=(0, 0, 0), font=font)
        if gt_pil is not None:
            x_gt = padding + (base_col + 1) * (img_w + padding)
            gt_x = x_gt + (img_w - _text_width(gt_label)) / 2
            draw.text((gt_x, label_y), gt_label, fill=(0, 0, 0), font=font)

    canvas.save(filepath)


def _save_individual_images(rec_images, gt_images, labels, cfg, has_gt=True):
    """Save each reconstructed (and ground-truth, if available) image as a separate PNG file.

    Files are written under ``reconstructions/individual/<data>_<model>_user<idx>/``
    with one subfolder per kind (``rec`` and ``gt``). When labels are available, the
    label is appended to the filename. The existing side-by-side grid is unaffected.
    """
    base_dir = os.path.join(
        "reconstructions",
        "individual",
        f"{cfg.case.data.name}_{cfg.case.model}_user{cfg.case.user.user_idx}",
    )
    rec_dir = os.path.join(base_dir, "rec")
    os.makedirs(rec_dir, exist_ok=True)
    if has_gt and gt_images is not None:
        gt_dir = os.path.join(base_dir, "gt")
        os.makedirs(gt_dir, exist_ok=True)

    n = len(rec_images)
    label_list = labels if isinstance(labels, list) else ([labels] * n if labels is not None else [None] * n)

    for idx in range(n):
        label_suffix = "" if label_list[idx] is None else f"_label{label_list[idx]}"
        rec_pil = torchvision.transforms.functional.to_pil_image(rec_images[idx])
        if os.path.exists(os.path.join(rec_dir, f"rec_{idx:04d}{label_suffix}.png")):
            # print(f"Warning: Overwriting existing reconstruction image at {os.path.join(rec_dir, f'rec_{idx:04d}{label_suffix}.png')}")
            count = len([f for f in os.listdir(rec_dir) if f.startswith(f"rec_{idx:04d}{label_suffix}")])
            rec_pil.save(os.path.join(rec_dir, f"rec_{idx:04d}{label_suffix}_{count+1}.png"))
        else:
            rec_pil.save(os.path.join(rec_dir, f"rec_{idx:04d}{label_suffix}.png"))
        if has_gt and gt_images is not None:
            gt_pil = torchvision.transforms.functional.to_pil_image(gt_images[idx])
            if os.path.exists(os.path.join(gt_dir, f"gt_{idx:04d}{label_suffix}.png")):
                # print(f"Warning: Overwriting existing GT image at {os.path.join(gt_dir, f'gt_{idx:04d}{label_suffix}.png')}")
                count = len([f for f in os.listdir(gt_dir) if f.startswith(f"gt_{idx:04d}{label_suffix}")])
                gt_pil.save(os.path.join(gt_dir, f"gt_{idx:04d}{label_suffix}_{count+1}.png"))
            else:
                gt_pil.save(os.path.join(gt_dir, f"gt_{idx:04d}{label_suffix}.png"))


def dump_metrics(cfg, metrics):
    """Simple yaml dump of metric values."""

    filepath = f"metrics_{cfg.case.data.name}_{cfg.case.model}_user{cfg.case.user.user_idx}.yaml"
    sanitized_metrics = dict()
    for metric, val in metrics.items():
        try:
            if torch.is_tensor(val) and val.device.type != "cpu":
                val = val.cpu()
            sanitized_metrics[metric] = np.asarray(val).item()
        except ValueError:
            sanitized_metrics[metric] = np.asarray(val).tolist()
    with open(filepath, "w") as yaml_file:
        yaml.dump(sanitized_metrics, yaml_file, default_flow_style=False)


def huggingface_offline_mode(huggingface_offline_mode):
    if huggingface_offline_mode:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
