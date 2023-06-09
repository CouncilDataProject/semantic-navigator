# [START cloudrun_semanticnavigator_dockerfile]
# [START run_semanticnavigator_dockerfile]

# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.10-slim

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# Install git
RUN apt-get -y update && apt-get -y install git

# Install dependencies
RUN pip install --no-cache-dir .

# Download / pre-cache model
RUN fetch-model-for-sem-nav

# Run the web server on startup
CMD semantic-navigator

# [END run_semanticnavigator_dockerfile]
# [END cloudrun_semanticnavigator_dockerfile]
