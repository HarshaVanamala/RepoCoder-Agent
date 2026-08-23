# Day 6: sandboxed test execution environment.
# Builds an image with Python + the requests repo's test dependencies
# pre-installed. At run time, test_runner.py mounts in a fresh copy of
# the repo (with a patch applied) so each test run is isolated and the
# image itself doesn't need rebuilding per patch.

FROM python:3.12-slim

WORKDIR /repo

# System deps some of requests' test dependencies (e.g. trustme) may need
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the full repo once at build time to prime the environment —
# requirements-dev.txt references "-e .[socks]" which needs the actual
# package present to install editable.
COPY data/test_repo /repo
RUN pip install --no-cache-dir -r requirements-dev.txt

ENTRYPOINT ["pytest"]