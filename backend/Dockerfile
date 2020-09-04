FROM python:3.7

# Set environment variables.
ENV VENV=/opt/venv
RUN python -m venv ${VENV}
ENV PATH="$VENV/bin:$PATH"

# Copy backend codes to docker.
COPY . /usr/src/backend
WORKDIR /usr/src/backend

# Install dependencies.
RUN pip install -r requirements.txt

# Open port.
EXPOSE 8000