FROM python:3.11

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt -i https://package-mirror.liara.ir/repository/pypi/simple --trusted-host package-mirror.liara.ir

# Copy the rest of the project
COPY ./src ./src

# Run the server
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]