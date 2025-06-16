.PHONY: install test lint run clean

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt

dev:
	@echo "Installing development dependencies..."
	pip install -r infrastructure/requirements/streamlit.txt
	pip install -e .

test:
	@echo "Running tests..."
	python -m pytest tests/ -v

lint:
	@echo "Running linting..."
	flake8 .

typecheck:
	@echo "Running type checking..."
	mypy .

run:
	@echo "Starting Streamlit app..."
	streamlit run app/streamlit_app.py

clean:
	@echo "Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache .mypy_cache .coverage coverage.xml htmlcov