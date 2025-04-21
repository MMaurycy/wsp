# syntax=docker/dockerfile:1

FROM python:3.10-slim                   # obraz bazowy

LABEL maintainer="Marcin Przybylski <mar.przybylski@o2.pl>"
LABEL description="Środowisko dla API i Dashboardu WSP"

WORKDIR /app                            # katalog roboczy

COPY requirements.txt ./               # najpierw zależności (dla cache)

# RUN apt-get update && apt-get install -y --no-install-recommends gcc \
#   && rm -rf /var/lib/apt/lists/*     # systemowe (opcjonalnie)

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .                                # reszta kodu

COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh         # prawa do uruchamiania

EXPOSE 8000 8501

ENTRYPOINT ["/app/entrypoint.sh"]       # start serwerów

# CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
