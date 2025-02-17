# Use an official Python runtime as a base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application code into the container
COPY . .

# Expose the port on which the app will run
EXPOSE 8000

# Command to run the FastAPI app using uvicorn (adjust the module path if needed)
CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "8000"]
