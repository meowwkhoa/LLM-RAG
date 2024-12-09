# Use the official Python base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt .

# Install the dependencies
RUN pip install -r requirements.txt --no-cache-dir

# Downgrade pydantic version by a seperate command
# since I don't want to install a lot of dependencies
# in requirement.txt again.
RUN pip install pydantic==1.10.13

# Copy the source code to the container
COPY src/*.py /app

# Expose the port that the FastAPI application will run on
EXPOSE 8001

# Start the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]