#!/bin/bash

sudo timedatectl set-timezone America/Chicago
curl -o Python.tar.xz https://www.python.org/ftp/python/3.14.4/Python-3.14.4.tar.xz
echo "d923c51303e38e249136fc1bdf3568d56ecb03214efdef48516176d3d7faaef8 Python.tar.xz" | sha256sum -c
tar -xvf Python.tar.xz

sudo apt update; sudo apt install -y libssl-dev libsqlite3-dev libffi-dev libbz2-dev liblzma-dev uuid-dev nvidia-driver-590 btop

export OPENSSL_DIR="/usr"
export LDFLAGS="-L$OPENSSL_DIR/lib"
export CPPFLAGS="-I$OPENSSL_DIR/include"

cd Python-3.14.4; \
	./configure --with-openssl="$OPENSSL_DIR" --with-openssl-rpath=auto --enable-optimizations --enable-loadable-sqlite-extensions --with-ensurepip=install; \
	make;
	sudo make altinstall

rm Python.tar.xz
rm -rf Python-3.14.4
sudo python3.14 -m pip install --root-user-action ignore -r ../requirements.txt