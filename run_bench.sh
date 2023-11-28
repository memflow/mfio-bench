#!/bin/sh

set -eE

mode="$1"

startdir="$PWD"

if [ "$mode" = "vagrant" ]; then
    cd vagrant

    vagrant up

    cd ..

    sudo screen -d -S latencyswitch -m ./latencyswitch.sh vagrant eth0

else
    if ! which docker-compose; then
        dcomp="podman-compose"
        iface="eth0"
        cont="docker_smb_1"
    else
        dcomp="docker-compose"
        iface="eth0"
        cont="docker-smb-1"
    fi

    cd docker

    mkdir -p vol

    if [ ! -f "vol/sample.img" ]; then
        dd if=/dev/urandom of=vol/sample.img bs=1G count=16
    fi

    "$dcomp" up --remove-orphans --force-recreate --build -d smb

    cd ..

    sudo screen -d -S latencyswitch -m ./latencyswitch.sh docker "$cont" "$iface"
fi

mkdir -p smb

echo "Mounting smb"

sleep 2

if [[ "$OSTYPE" == "darwin"* ]]; then
    while ! mount -t smbfs //guest@127.0.0.1:4445/main ./smb; do
        sleep 1
    done
else
    while ! sudo mount -t cifs -o port=4445,user=guest,password="" //127.0.0.1/main ./smb; do
        sleep 1
    done
fi

echo "Mounted smb"

exit_handler() {
    set +e
    echo "Exit handler"
    echo "bench-quit" | nc localhost 12345 -v
    echo "$PWD"

    if [ "$mode" = "vagrant" ]; then
        cd "${startdir}/vagrant"

        vagrant halt
    else
        cd "${startdir}/docker"

        "$dcomp" down
    fi
    exit
}

trap exit_handler SIGTERM SIGQUIT ERR EXIT

cargo +nightly run --release -- --bench latency_mode

