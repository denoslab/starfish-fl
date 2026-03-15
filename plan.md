# FederatedUNet Execution Plan

## Sub-tasks
1. Validate local Python environment for `image_segmentation_model`.
2. Check whether a virtual environment already exists; create one only if missing.
3. Install/verify dependencies required by `federated_train.py` inside that virtual environment.
4. Run a local smoke test of federated segmentation (`--num_clients 2 --num_rounds 1 --local_epochs 1`).
5. Capture runtime errors, then patch minimal fixes needed for local execution.
6. Re-run smoke test until the script starts training successfully.
7. Begin Starfish integration steps from approved plan (`router` limits, controller task/files, Docker/GPU config, test data script, docs).

## Current focus
- Execute sub-tasks 1-4 first.
- Use virtual environment for all local runs.

## Progress update (2026-03-13)
- Sub-task 1 completed: validated local Python environment (`.venv`, Python 3.10.10).
- Sub-task 2 completed: existing virtual environment reused.
- Sub-task 3 completed: dependencies aligned to a consistent GPU-capable TensorFlow stack in venv.
- Sub-task 4 completed with bare minimum data:
	- Command: `"/home/farhan/Work/DENOS Lab/image_segmentation_model/.venv/bin/python" federated_train.py --num_clients 2 --num_rounds 1 --local_epochs 1 --batch_size 1 --max_samples_per_client 1 --max_test_samples 1 --base_dir "/home/farhan/Work/DENOS Lab/image_segmentation_model/"`
	- Result: run completed end-to-end.

## GPU-only verification details
- Initial issue: TensorFlow/CUDA package mismatch caused CPU fallback and import errors.
- Applied venv package alignment:
	- `tensorflow==2.15.1`
	- `keras==2.15.0`
	- `tf_keras==2.15.0`
	- `numpy==1.26.4`
- Verified GPU visibility in venv (`GPU_COUNT 1`).
- Verified training workers used GPU from logs:
	- `[Client 0] GPU available`
	- `[Client 1] GPU available`

## Next step
- Proceed with Starfish integration run path, now that local GPU-backed behavior is confirmed.

## Starfish integration progress (2026-03-14)
- Implemented FederatedUNet task at:
	- `controller/starfish/controller/tasks/federated_unet/task.py`
- Added dataset/artifact helper support for segmentation zip + binary weights at:
	- `controller/starfish/controller/file/file_utils.py`
- Added dedicated UNet dependency lock for controller container pip install:
	- `controller/requirements-unet.txt`
- Updated controller image build to install UNet requirements via pip while keeping base image unchanged:
	- `controller/Dockerfile`
- Added router upload size limits for larger segmentation artifacts/datasets:
	- `router/starfish/settings.py`
- Added FederatedUNet config and dataset format documentation:
	- `controller/TASK_GUIDE.md`
- Added minimal test zip generator for two-site local tests:
	- `workbench/prepare_test_data.py`

## Container validation result
- Built image: `starfish-controller-unet-test`
- Verified inside container:
	- `tensorflow==2.15.1`
	- `segmentation-models==1.0.1`
	- `FederatedUNet` task import succeeds
	- UNet model build (`sm.Unet('resnet50', ...)`) succeeds

## Container GPU test status (2026-03-14)
- Used existing controller image base (`python:3.10`) and ran exact command in container:
	- `federated_train.py --num_clients 2 --num_rounds 1 --local_epochs 1 --batch_size 1 --max_samples_per_client 1 --max_test_samples 1`
- Command was executed with `--gpus all` and mounted dataset path.
- Resolution:
	- Enabled NVIDIA Container Toolkit and switched to `default` docker context for GPU runtime.
	- Added CUDA runtime wheel deps and `matplotlib` to `controller/requirements-unet.txt`.
- Final result (default context):
	- Container run completed end-to-end with GPU client training:
		- `[Client 0] GPU available`
		- `[Client 1] GPU available`
	- Evaluation step also completed, including ROC plot + metrics output.

## Remaining integration run-path work
- Run full Starfish two-site project flow with `FederatedUNet` task JSON and dataset uploads.
- Validate run state transitions through `SUCCESS` for round 1.