# ---- Base: PyTorch + CUDA runtime ----
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel AS builder

WORKDIR /app

# System deps for building CUDA extensions and open3d
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc-9 \
    g++-9 \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir ninja && \
    pip install --no-cache-dir -e "." 2>/dev/null || \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Pre-compile CUDA extensions (JIT cache lives in /root/.cache/torch_extensions)
RUN python -c "from pcdiff.modules.functional.backend import _backend; print('CUDA extensions compiled')"

# ---- Runtime: slimmer image ----
FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages and compiled extensions from builder
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /root/.cache/torch_extensions /root/.cache/torch_extensions
COPY --from=builder /app /app

# Default env-var overrides (users can set at runtime)
ENV PCDIFF_SKULLFIX_RESULTS=datasets/SkullFix/results/syn \
    PCDIFF_SKULLBREAK_RESULTS=datasets/SkullBreak/results/syn

ENTRYPOINT ["python"]
