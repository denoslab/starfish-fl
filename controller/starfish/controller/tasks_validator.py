from celery.utils.log import get_task_logger

from starfish.controller.utils import parse_tasks, load_class, camel_to_snake

logger = get_task_logger(__name__)


class TaskValidator:
    errors = None
    tasks = None

    def __init__(self, task_str):
        self.tasks = parse_tasks(task_str)
        self.errors = []

    def get_error_msg(self):
        if not self.is_valid():
            return self.errors[0]

    def is_valid(self):
        return len(self.errors) == 0

    def get_validated_tasks(self):
        self.pre_validate()
        self.validate_base_info()
        self.post_validate_tasks()
        if self.is_valid():
            return self.tasks
        return None

    def pre_validate(self):
        if self.is_valid() and (self.tasks is None or len(self.tasks) == 0):
            self.errors.append('At least one task provided ')

    def validate_base_info(self):
        if self.is_valid():
            for task in self.tasks:
                self.validate_keys(task)
                self.validate_seq(task)
                self.validate_model(task)
                self.validate_config(task)

    def validate_keys(self, task):
        if self.is_valid():
            if 'seq' not in task or 'model' not in task or 'config' not in task:
                self.errors.append(
                    'Valid task must contain seq , model and config')

    def post_validate_tasks(self):
        if self.is_valid():
            count = len(self.tasks)
            for idx, task in enumerate(self.tasks):
                if idx == 0 and task['seq'] != 1:
                    self.errors.append('seq should start with 1')
                if idx < count - 1:
                    next_task = self.tasks[idx + 1]
                    if next_task['seq'] != task['seq'] + 1:
                        self.errors.append('seq of tasks is not consecutive')

    def validate_seq(self, task):
        if self.is_valid():
            seq = task['seq']
            if seq is None or not isinstance(seq, int) or seq < 0:
                self.errors.append('seq could must be non-negative int value')

    def validate_model(self, task):
        if self.is_valid():
            model = task['model']
            model_class = None
            snake_name = camel_to_snake(model)
            # Try loading from main tasks module first, then stats_models subpackage
            module_paths = [
                'starfish.controller.tasks.{}'.format(snake_name),
                'starfish.controller.tasks.stats_models.{}'.format(snake_name)
            ]
            for module_path in module_paths:
                try:
                    model_class = load_class(module_path, model)
                    if model_class is not None:
                        break
                except (ImportError, AttributeError):
                    continue
            if model_class is None:
                logger.warn("{} not found in any module path".format(model))
                self.errors.append(
                    'model corresponding task could not be found')

    def validate_config(self, task):
        if self.is_valid():
            config = task['config']
            if config is None or not isinstance(config, dict) or len(config) == 0:
                self.errors.append(
                    'config must be a key-value map and could not be empty')
