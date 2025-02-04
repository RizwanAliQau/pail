# Use the official Python image from the Docker Hub
FROM python:3.10-slim
RUN apt-get update && apt-get install -y gcc
RUN apt-get install -y ffmpeg libsm6 libxext6
# Set the working directory in the container
WORKDIR /pai_lib

# Copy the requirements.txt file into the container at /app
COPY requirement.txt .

# Install any necessary dependencies
RUN pip install --no-cache-dir -r requirement.txt

# Copy the rest of the working directory contents into the container at /pai_lib
COPY . .

# Specify the command to run on container start
# CMD ["python", "__.py"]
