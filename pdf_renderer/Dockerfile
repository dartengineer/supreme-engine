# HTML -> PDF renderer for the CV generator (WeasyPrint needs Pango/GTK, which are easy on Linux).
# Build:  docker build -t cv-pdf-renderer pdf_renderer
# Run:    docker run -d --name cv-pdf --restart unless-stopped -p 127.0.0.1:5001:5001 cv-pdf-renderer
FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz-subset0 \
        fonts-liberation2 fonts-dejavu-core fonts-open-sans \
    && rm -rf /var/lib/apt/lists/*

# pydyf is pinned: WeasyPrint 62.x breaks with pydyf 0.11+ ("'super' object has no attribute 'transform'").
RUN pip install --no-cache-dir weasyprint==62.3 pydyf==0.10.0

RUN useradd --create-home renderer
USER renderer
WORKDIR /home/renderer
COPY server.py .

EXPOSE 5001
CMD ["python", "server.py"]
