Development Setup
For development, install additional requirements:

bash
pip install -r infrastructure/requirements/streamlit.txt
pip install -e .
Running the Application
bash
make run
This will start the Streamlit web application on port 8501.

text

### 3. scripts/setup_environment.sh
```bash
#!/bin/bash

# Setup script for the data preprocessing pipeline

# Exit on error
set -e

echo "Setting up Python virtual environment..."
python -m venv venv
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements..."
pip install -r requirements.txt

if [ "$1" = "dev" ]; then
    echo "Installing development requirements..."
    pip install -r infrastructure/requirements/streamlit.txt
    pip install -e .
fi

echo "Setting up environment variables..."
cp .env.example .env

echo "Setup complete!"
echo "Activate the virtual environment with: source venv/bin/activate"