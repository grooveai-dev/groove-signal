FROM python:3.12-slim

RUN useradd --create-home --shell /bin/bash groove

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chown -R groove:groove /app
USER groove

EXPOSE 8770 8771

CMD ["python", "-m", "src.signal.server"]
