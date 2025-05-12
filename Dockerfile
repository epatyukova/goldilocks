# Use official Python base image
FROM python:3.11-slim

# Set environment variables
ENV POETRY_VERSION=1.8.2 \
    POETRY_NO_INTERACTION=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y curl build-essential

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

# Copy project files
COPY . .

# Install dependencies via Poetry
RUN poetry install --no-root

# Expose Streamlit default port
EXPOSE 8501

# Run the app
CMD ["poetry", "run", "streamlit", "run", "src/qe_input/QE_input.py", "--server.headless=true", "--server.port=8501", "--server.enableCORS=false"]
