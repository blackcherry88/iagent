build:
	nbdev_prepare
	poetry build

format:
	poetry run black .
	poetry run isort .
	poetry run ruff check --fix .
