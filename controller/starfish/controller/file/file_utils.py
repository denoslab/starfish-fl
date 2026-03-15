import logging
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

base_folder = '/starfish-controller/local'

logs_name = 'logs.txt'

mid_artifacts_name = 'mid-artifacts'

mid_artifacts_dir = 'all-mid-artifacts'

artifacts_name = 'artifacts'

dataset_name = 'dataset'


def create_if_not_exist(url):
    if url:
        try:
            file_path = Path(url)
            if not file_path.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.touch()
        except Exception as e:
            logger.warning(
                "Error while creating log file: {} due to".format(e))


def gen_url(run_id, task_seq, round_seq, file_name=None):
    if not run_id or not task_seq or not round_seq:
        return None
    if file_name:
        return f"{base_folder}/{run_id}/{task_seq}/{round_seq}/{file_name}"
    else:
        return f"{base_folder}/{run_id}/{task_seq}/{round_seq}"


def gen_dataset_url(run_id):
    if not run_id:
        return None
    return f"{base_folder}/{run_id}/"


def gen_logs_url(run_id, task_seq, round_seq):
    return gen_url(run_id, task_seq, round_seq, file_name=logs_name)


def gen_artifacts_url(run_id, task_seq, round_seq):
    return gen_url(run_id, task_seq, round_seq, file_name=artifacts_name)


def gen_mid_artifacts_url(run_id, task_seq, round_seq):
    return gen_url(run_id, task_seq, round_seq, file_name=mid_artifacts_name)


def gen_binary_mid_artifacts_url(run_id, task_seq, round_seq):
    return gen_mid_artifacts_url(run_id, task_seq, round_seq)


def gen_all_mid_artifacts_url(project_id, batch):
    if not project_id or not batch:
        return None
    return f"{base_folder}/{mid_artifacts_dir}/{project_id}/{batch}/"


def downloaded_artifacts_url(run_id, task_seq, round_seq):
    if not run_id or not task_seq or not round_seq:
        return None
    return f"{base_folder}/{artifacts_name}/{run_id}/{task_seq}/{round_seq}/"


def gen_binary_artifacts_url(run_id, task_seq, round_seq):
    return gen_artifacts_url(run_id, task_seq, round_seq)


def download_all_mid_artifacts(project_id, batch, content):
    dir_url = gen_all_mid_artifacts_url(project_id, batch)
    if dir_url:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(content)
        file_path = Path(dir_url)
        file_path.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(temp_file.name, 'r') as zip_ref:
            zip_ref.extractall(file_path)
        return file_path.absolute()
    return None


def download_artifacts(run_id, task_seq, round_seq, content):
    dir_url = downloaded_artifacts_url(run_id, task_seq, round_seq)
    if dir_url:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(content)
        file_path = Path(dir_url)
        file_path.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(temp_file.name, 'r') as zip_ref:
            zip_ref.extractall(file_path)
        return file_path.absolute()
    return None


def read_file_from_url(url):
    if url:
        try:
            file_obj = open(url, 'r')
            return file_obj
        except FileNotFoundError:
            logger.warning("File not found at {}".format(url))
            return None
    return None


def read_binary_file_from_url(url):
    if url:
        try:
            file_obj = open(url, 'rb')
            return file_obj
        except FileNotFoundError:
            logger.warning("File not found at {}".format(url))
            return None
    return None


def load_dataset_by_run(run_id):
    combined_csv_file = read_file_from_url(gen_dataset_url(run_id) + 'dataset')
    if combined_csv_file:
        try:
            combined_data = pd.read_csv(combined_csv_file, header=None)
            X = combined_data.iloc[:, :-1].values
            y = combined_data.iloc[:, -1].values
            return X, y
        except Exception as e:
            logger.warning("Failed to read data set due to {}".format(e))
    return None, None


def _is_supported_image(file_name: str) -> bool:
    return file_name.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))


def load_image_dataset_by_run(run_id, patch_size):
    dataset_zip_path = gen_dataset_url(run_id) + dataset_name
    try:
        from PIL import Image
    except Exception as e:
        logger.warning("Pillow is not available due to {}".format(e))
        return None, None

    try:
        with zipfile.ZipFile(dataset_zip_path, 'r') as zf:
            images = sorted([
                name for name in zf.namelist()
                if name.startswith('images/') and _is_supported_image(name)
            ])
            masks = sorted([
                name for name in zf.namelist()
                if name.startswith('masks/') and _is_supported_image(name)
            ])

            if not images or not masks:
                logger.warning("Dataset zip missing images/ or masks/ content")
                return None, None

            if len(images) != len(masks):
                logger.warning("Image/mask count mismatch: {} vs {}".format(len(images), len(masks)))
                return None, None

            image_names = [Path(p).name for p in images]
            mask_names = [Path(p).name for p in masks]
            if image_names != mask_names:
                logger.warning("Image/mask filenames do not align after sorting")
                return None, None

            image_stack = []
            mask_stack = []
            for image_entry, mask_entry in zip(images, masks):
                with zf.open(image_entry) as image_fp, zf.open(mask_entry) as mask_fp:
                    image = Image.open(image_fp).convert('L').resize((patch_size, patch_size), Image.BILINEAR)
                    mask = Image.open(mask_fp).convert('L').resize((patch_size, patch_size), Image.NEAREST)

                    image_arr = np.asarray(image, dtype=np.float32) / 255.0
                    mask_arr = (np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32)

                    image_stack.append(image_arr)
                    mask_stack.append(mask_arr)

            return np.asarray(image_stack, dtype=np.float32), np.asarray(mask_stack, dtype=np.float32)

    except FileNotFoundError:
        logger.warning("Dataset zip not found at {}".format(dataset_zip_path))
    except zipfile.BadZipFile:
        logger.warning("Dataset at {} is not a valid zip file".format(dataset_zip_path))
    except Exception as e:
        logger.warning("Failed to load image dataset due to {}".format(e))

    return None, None
