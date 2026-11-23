"""A task's logger writes the round's log file and prints to the console only in a
process without logging of its own."""

import io
import logging
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from starfish.controller.tasks import abstract_task


class Task(abstract_task.AbstractTask):
    agents_allowed = False

    def validate(self):
        return True

    def prepare_data(self):
        return True

    def training(self):
        return True

    def do_aggregate(self):
        return True


class Collect(logging.Handler):

    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def run(run_id):
    return {'project': 1, 'id': run_id, 'batch': 1, 'role': 'PA', 'status': 'standby',
            'cur_seq': 1, 'tasks': [{'model': 'X', 'config': {'current_round': 1}}]}


class TaskLoggerTest(TestCase):

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)

    def make_task(self, run_id, root_handlers):
        url = self.folder / 'logs-{}.txt'.format(run_id)
        console = io.StringIO()
        with patch.object(abstract_task, 'gen_logs_url', return_value=str(url)), \
                patch.object(logging.getLogger(), 'handlers', root_handlers), \
                patch('sys.stderr', console):
            task = Task(run(run_id))
            task.logger.info('Round 1 starts')
        self.addCleanup(self.close, task.logger)
        return task, url.read_text(), console.getvalue()

    @staticmethod
    def close(logger):
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

    def test_a_process_with_root_logging_prints_each_line_once(self):
        root = Collect()
        task, log_file, console = self.make_task(9101, [root])
        self.assertEqual([type(h) for h in task.logger.handlers], [logging.FileHandler])
        self.assertEqual(root.messages.count('Round 1 starts'), 1)
        self.assertEqual(console, '')
        self.assertIn('[INFO] Round 1 starts', log_file)

    def test_a_process_without_logging_prints_to_the_console(self):
        task, log_file, console = self.make_task(9102, [])
        self.assertEqual([type(h) for h in task.logger.handlers],
                         [logging.FileHandler, logging.StreamHandler])
        self.assertEqual(console.count('[INFO] Round 1 starts'), 1)
        self.assertIn('[INFO] Round 1 starts', log_file)
