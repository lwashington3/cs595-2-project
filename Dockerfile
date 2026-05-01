FROM python:3.14.3-slim AS base

WORKDIR /build

COPY requirements.txt ./

CMD ["python3", "-m", "washLTH"]

RUN pip install --upgrade pip; \
	pip install --root-user-action ignore --no-cache-dir -r /build/requirements.txt; \
	rm /build/requirements.txt; \
	apt update && \
	apt install -y --no-install-recommends texlive-full texlive-latex-extra texlive-fonts-recommended latexmk biber && \
    rm -rf /var/lib/apt/lists/*

COPY ./scripts /lth/scripts
COPY ./washLTH /usr/local/lib/python3.14/site-packages/washLTH

WORKDIR /lth