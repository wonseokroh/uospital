FROM node:12

# Copy frontend codes to docker.
COPY . /usr/src/frontend
WORKDIR /usr/src/frontend

# Install dependencies.
RUN yarn

# Build.
RUN yarn build

# Open port.
EXPOSE 8080