# Contributing

Contributions are welcome! Please follow these guidelines.

## Guidelines

1. Code follows existing style and conventions
2. Tests are included for new features
3. Documentation is updated as needed
4. Docker configurations are tested

## Code Formatting

Format Python code using autopep8:

```bash
autopep8 --exclude='*/migrations/*' --in-place --recursive ./starfish/
```

## Running Tests

```bash
# Router
cd router && python3 manage.py test

# Controller
cd controller && python3 manage.py test

# Via Docker
docker exec -it starfish-router poetry run python3 manage.py test
docker exec -it starfish-controller poetry run python3 manage.py test
```

## Adding a New Task

See [Architecture](architecture.md#adding-a-new-ml-task) for how to add Python and R tasks.

For every new task:

1. Implement the task class with `prepare_data()`, `training()`, and `do_aggregate()`
2. Add test coverage in `controller/starfish/controller/test_<task_name>.py`
3. Add configuration docs to `controller/TASK_GUIDE.md`
4. Update `controller/USER_GUIDE.md` with result interpretation
5. Update the supported tasks tables in `README.md` and `docs/index.md`

## License

Apache 2.0

## Support

For issues, questions, or contributions, [open an issue](https://github.com/denoslab/starfish-fl/issues) in the repository.
