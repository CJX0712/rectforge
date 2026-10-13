FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rectforge/ ./rectforge/
COPY examples/ ./examples/
COPY tests/ ./tests/
COPY pyproject.toml .

# 确定性：容器内单线程 BLAS
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    BLIS_NUM_THREADS=1

CMD ["python", "examples/run_demo.py"]
