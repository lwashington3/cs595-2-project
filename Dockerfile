FROM python:3.14.3-slim AS base

WORKDIR /build

ENV PATH="$PATH:/lth/scripts", HISTTIMEFORMAT="%F %T "
# /lth/ablation script
CMD ["ablation"]
#CMD ["python3", "-m", "washLTH"]

COPY requirements.txt /build/requirements.txt
COPY ./scripts /lth/scripts

RUN pip install --upgrade pip; \
	pip install --root-user-action ignore --no-cache-dir -r /build/requirements.txt; \
	rm -rf /build; \
	apt update && \
	apt install -y --no-install-recommends texlive-full texlive-latex-extra texlive-fonts-recommended latexmk biber && \
    rm -rf /var/lib/apt/lists/*; \
    chmod -R +x /lth/scripts

COPY ./washLTH /usr/local/lib/python3.14/site-packages/washLTH

WORKDIR /lth