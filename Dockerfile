FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH=/workspace/DEYOLO

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    zip \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    tmux \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install "setuptools==69.5.1" wheel

RUN pip install \
    "numpy==1.26.4" \
    requests \
    einops \
    wandb \
    tqdm \
    pyyaml \
    matplotlib \
    scipy \
    opencv-python \
    Pillow \
    seaborn \
    pandas \
    psutil \
    py-cpuinfo \
    thop \
    ipykernel \
    kaggle \
    tensorboard \
    dill \
    jupyterlab

RUN git clone https://github.com/chips96/DEYOLO.git /workspace/DEYOLO

RUN sed -i 's/np\.trapz/np.trapezoid/g' /workspace/DEYOLO/ultralytics/yolo/utils/metrics.py

RUN sed -i "s/return torch.load(file, map_location='cpu'), file/return torch.load(file, map_location='cpu', weights_only=False), file/g" /workspace/DEYOLO/ultralytics/nn/tasks.py

RUN echo '# Ultralytics YOLO' > /workspace/DEYOLO/ultralytics/yolo/utils/callbacks/raytune.py && \
    echo 'try:' >> /workspace/DEYOLO/ultralytics/yolo/utils/callbacks/raytune.py && \
    echo '    import ray' >> /workspace/DEYOLO/ultralytics/yolo/utils/callbacks/raytune.py && \
    echo '    from ray import tune' >> /workspace/DEYOLO/ultralytics/yolo/utils/callbacks/raytune.py && \
    echo '    TUNE_AVAILABLE = True' >> /workspace/DEYOLO/ultralytics/yolo/utils/callbacks/raytune.py && \
    echo 'except (ImportError, AssertionError):' >> /workspace/DEYOLO/ultralytics/yolo/utils/callbacks/raytune.py && \
    echo '    tune = None' >> /workspace/DEYOLO/ultralytics/yolo/utils/callbacks/raytune.py && \
    echo '    TUNE_AVAILABLE = False' >> /workspace/DEYOLO/ultralytics/yolo/utils/callbacks/raytune.py && \
    echo '' >> /workspace/DEYOLO/ultralytics/yolo/utils/callbacks/raytune.py && \
    echo 'def on_fit_epoch_end(trainer):' >> /workspace/DEYOLO/ultralytics/yolo/utils/callbacks/raytune.py && \
    echo '    if not TUNE_AVAILABLE or tune is None:' >> /workspace/DEYOLO/ultralytics/yolo/utils/callbacks/raytune.py && \
    echo '        return' >> /workspace/DEYOLO/ultralytics/yolo/utils/callbacks/raytune.py && \
    echo '    try:' >> /workspace/DEYOLO/ultralytics/yolo/utils/callbacks/raytune.py && \
    echo '        metrics = trainer.metrics' >> /workspace/DEYOLO/ultralytics/yolo/utils/callbacks/raytune.py && \
    echo '        metrics["epoch"] = trainer.epoch' >> /workspace/DEYOLO/ultralytics/yolo/utils/callbacks/raytune.py && \
    echo '    except Exception:' >> /workspace/DEYOLO/ultralytics/yolo/utils/callbacks/raytune.py && \
    echo '        pass' >> /workspace/DEYOLO/ultralytics/yolo/utils/callbacks/raytune.py && \
    echo '' >> /workspace/DEYOLO/ultralytics/yolo/utils/callbacks/raytune.py && \
    echo 'callbacks = {"on_fit_epoch_end": on_fit_epoch_end} if TUNE_AVAILABLE else {}' >> /workspace/DEYOLO/ultralytics/yolo/utils/callbacks/raytune.py

RUN python3 -m ipykernel install --name=python3 --display-name "Python 3 (DEYOLO)"

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=", "--NotebookApp.password=", "--ServerApp.iopub_data_rate_limit=10000000"]