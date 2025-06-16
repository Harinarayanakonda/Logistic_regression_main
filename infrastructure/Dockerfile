# Stage 1: Build Stage - Install dependencies and build any necessary assets
FROM python:3.9-slim-buster AS build_stage

WORKDIR /app

# Install system dependencies for pandas/numpy (e.g., for excel files)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files first to leverage Docker cache
COPY requirements/base.txt requirements/base.txt
COPY requirements/streamlit.txt requirements/streamlit.txt
COPY requirements.txt requirements.txt

# Install base requirements
RUN pip install --no-cache-dir -r requirements/base.txt

# Install streamlit requirements
RUN pip install --no-cache-dir -r requirements/streamlit.txt

# Install main project requirements (if different from base/streamlit)
# If requirements.txt is a superset, this might be redundant.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Stage 2: Production Stage - Create a smaller image for deployment
FROM python:3.9-slim-buster AS production_stage

WORKDIR /app

# Copy only the installed packages and application code from the build stage
COPY --from=build_stage /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=build_stage /usr/local/bin /usr/local/bin
COPY --from=build_stage /app /app

# Expose the port Streamlit runs on
EXPOSE 8501

# Set environment variables for Streamlit (optional, but good practice)
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_LOGGER_LEVEL=info

# Command to run the Streamlit application
ENTRYPOINT ["streamlit", "run", "src/interfaces/streamlit_app/app.py", "--server.port=8501", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]