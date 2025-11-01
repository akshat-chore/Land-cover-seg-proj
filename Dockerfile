# Base image
FROM python:3.9

# Set working directory in container
WORKDIR /segment_project

# Copy dependencies file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY ./config ./config
COPY ./data ./data
COPY ./models ./models
COPY ./src ./src

# Run Streamlit app on container start
CMD ["streamlit", "run", "./src/app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
