import json
import logging
import os
import subprocess
import tempfile
from abc import ABC
from pathlib import Path

from starfish.controller.file.file_utils import (
    gen_mid_artifacts_url, gen_all_mid_artifacts_url, gen_artifacts_url,
    downloaded_artifacts_url, gen_dataset_url
)
from starfish.controller.tasks.abstract_task import AbstractTask


class AbstractRTask(AbstractTask, ABC):
    """
    Base class for FL tasks whose core logic is written in R.

    Subclasses must set ``r_script_dir`` to the directory containing
    ``prepare_data.R``, ``training.R``, and ``aggregate.R``.
    """
    r_script_dir: str = None  # set by subclass

    def __init__(self, run):
        super().__init__(run)
        self.sample_size = None

    # ------------------------------------------------------------------
    # R bridge helpers
    # ------------------------------------------------------------------

    def _run_r_script(self, script_name, input_json_path, output_json_path):
        """Run an R script via ``Rscript`` and return True on success."""
        script_path = os.path.join(self.r_script_dir, script_name)
        cmd = ['Rscript', '--vanilla', script_path,
               input_json_path, output_json_path]
        self.logger.debug("Running R script: {}".format(' '.join(cmd)))
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600)
        if result.stdout:
            self.logger.debug("R stdout: {}".format(result.stdout))
        if result.stderr:
            self.logger.debug("R stderr: {}".format(result.stderr))
        if result.returncode != 0:
            self.logger.warning(
                "R script {} failed with exit code {}: {}".format(
                    script_name, result.returncode,
                    result.stderr.strip() if result.stderr else '(no stderr)'))
            return False
        return True

    def _write_input_json(self, extra=None):
        """Write input JSON for an R script and return the file path."""
        data = {
            'run_id': self.run_id,
            'project_id': self.project_id,
            'batch_id': self.batch_id,
            'cur_seq': self.cur_seq,
            'round': self.get_round(),
            'config': self.tasks[self.cur_seq - 1].get('config', {}),
        }
        if extra:
            data.update(extra)
        fd, path = tempfile.mkstemp(suffix='.json', prefix='r_input_')
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f)
        return path

    def _make_output_path(self):
        fd, path = tempfile.mkstemp(suffix='.json', prefix='r_output_')
        os.close(fd)
        return path

    def _read_output_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # AbstractTask interface
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        task_round = self.get_round()
        self.logger.debug(
            "Run {} - task {} - round {} task begins".format(
                self.run_id, self.cur_seq, task_round))
        return self.download_artifact()

    def prepare_data(self) -> bool:
        self.logger.debug(
            'Loading dataset for run {} ...'.format(self.run_id))

        dataset_url = gen_dataset_url(self.run_id)
        data_path = dataset_url + 'dataset' if dataset_url else None
        if not data_path or not os.path.exists(data_path):
            self.logger.warning("Dataset not found at {}".format(data_path))
            return False

        extra = {'data_path': data_path}

        # Load previous model if not first round
        if not self.is_first_round():
            previous_model = self._load_previous_model()
            if previous_model:
                extra['previous_model'] = previous_model

        input_path = self._write_input_json(extra)
        output_path = self._make_output_path()

        try:
            if not self._run_r_script('prepare_data.R', input_path, output_path):
                return False
            result = self._read_output_json(output_path)
            self.sample_size = result.get('sample_size', 0)
            return result.get('valid', False)
        finally:
            self._cleanup_temp(input_path, output_path)

    def training(self) -> bool:
        self.logger.info('Starting R training...')

        dataset_url = gen_dataset_url(self.run_id)
        data_path = dataset_url + 'dataset' if dataset_url else None

        extra = {'data_path': data_path, 'sample_size': self.sample_size}

        if not self.is_first_round():
            previous_model = self._load_previous_model()
            if previous_model:
                extra['previous_model'] = previous_model

        input_path = self._write_input_json(extra)
        output_path = self._make_output_path()

        try:
            if not self._run_r_script('training.R', input_path, output_path):
                return False
            result = self._read_output_json(output_path)
            self.logger.info('R training complete.')
            url = gen_mid_artifacts_url(
                self.run_id, self.cur_seq, self.get_round())
            self.logger.info(
                "Upload: {} \n to: {}".format(result, url))
            return self.save_artifacts(url, json.dumps(result))
        finally:
            self._cleanup_temp(input_path, output_path)

    def do_aggregate(self) -> bool:
        mid_artifacts = self._collect_mid_artifacts()
        if not mid_artifacts:
            self.logger.warning(
                "No mid-artifacts found for aggregation")
            return False

        extra = {'mid_artifacts': mid_artifacts}
        input_path = self._write_input_json(extra)
        output_path = self._make_output_path()

        try:
            if not self._run_r_script('aggregate.R', input_path, output_path):
                return False
            result = self._read_output_json(output_path)
            self.sample_size = result.get('sample_size', 0)
            url = gen_artifacts_url(
                self.run_id, self.cur_seq, self.get_round())
            self.logger.info(
                "Upload: {} \n to: {}".format(result, url))
            if self.save_artifacts(url, json.dumps(result)):
                self.upload(True)
                return True
            return False
        finally:
            self._cleanup_temp(input_path, output_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_previous_model(self):
        seq_no, round_no = self.get_previous_seq_and_round()
        if not seq_no or not round_no:
            return None
        directory = downloaded_artifacts_url(self.run_id, seq_no, round_no)
        if not directory:
            return None
        for path in Path(directory).rglob(
                "*-{}-{}-artifacts".format(seq_no, round_no)):
            with open(str(path), 'r') as f:
                for line in f:
                    return json.loads(line)
        return None

    def _collect_mid_artifacts(self):
        artifacts = []
        directory = gen_all_mid_artifacts_url(self.project_id, self.batch_id)
        if not directory:
            return artifacts
        for path in Path(directory).rglob(
                "*-{}-{}-mid-artifacts".format(self.cur_seq, self.get_round())):
            with open(str(path), 'r') as f:
                for line in f:
                    artifacts.append(json.loads(line))
        return artifacts

    def _cleanup_temp(self, *paths):
        for p in paths:
            try:
                os.unlink(p)
            except OSError:
                pass
