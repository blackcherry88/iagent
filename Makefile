build:
	nbdev_prepare
	poetry run black .
	poetry run isort .
	poetry run ruff check .
	poetry build
