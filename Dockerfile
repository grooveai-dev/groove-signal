FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# WebSocket + HTTP ports. TLS is typically terminated upstream at nginx.
EXPOSE 8770 8771

CMD ["python", "-m", "src.signal.server"]
