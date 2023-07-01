#!/bin/sh

set -e

screen -d -S latencyswitch -m ./latencyswitch.sh

cd docker

docker-compose up --remove-orphans --force-recreate --build -d smb

cd ..

mkdir -p smb

echo "Mounting smb"

sleep 2

while ! mount -t smbfs //guest@127.0.0.1:4445/main ./smb; do
	sleep 1
done

echo "Mounted smb"

cargo run --release -- --bench

pkill -P $(screen -ls | grep latencyswitch | cut -d. -f1)

umount smb

cd docker

docker-compose down
