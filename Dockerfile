# Use a lightweight and optimized base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy necessary files
COPY requirements.txt ./
COPY backend/ ./backend/
COPY models/ ./models/


# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the API port (8000)
EXPOSE 8000

# Command to start FastAPI with Uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]