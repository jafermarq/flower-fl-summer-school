FROM flwr/supernode:latest

# Install basic dependencies (with CPU support only)
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install datasets==3.1.0

# Copy data into the image
# This will be the data the SuperNode has access to
ARG DATA_DIR
COPY --chown=app:app ${DATA_DIR} /app/data

# Launch Flower SuperNode and connect it to the SuperLink in the MegaDataGrid
# Point it to the directory where the data is (so the ClientApp can use it easily)
ENTRYPOINT ["flower-supernode", "--superlink", "megadata-fleet.flower.ai:443", "--node-config", "path-to-data='/app/data'"]