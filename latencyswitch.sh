#!/bin/sh

prev=0

start_dir="$(pwd)"
ourpid="$$"

main_loop() {
    iface="$1"
    prefix="$2"
    while read line; do
        >&2 echo umount "${start_dir}/smb"
        if [ "$line" = "bench-quit" ]; then
            for i in $(seq 0 10); do
                if >&2 umount "${start_dir}/smb"; then
                    >&2 pkill -P "$ourpid" > /dev/null
                fi
                sleep 0.5
            done
            >&2 umount -l "${start_dir}/smb"
            >&2 pkill -P "$ourpid" > /dev/null
            break
        fi
        >&2 echo "${line}ms"

        if [ "$prev" -ne "0" ]; then
            echo "$prefix" tc qdisc del dev $iface root netem
        fi

        if [ "$line" -ne "0" ]; then
            echo "$prefix" tc qdisc add dev $iface root netem delay "${line}ms"
        fi

        prev="$line"
    done
}


socat - TCP-LISTEN:12345,fork,reuseaddr | \
    if [ "$1" = "vagrant" ]; then
        cd vagrant
        main_loop "$2" "sudo" | sudo -u "$SUDO_USER" vagrant ssh
    else
        main_loop "$2" "" | sudo -u "$SUDO_USER" docker exec -i "$3" /bin/sh
    fi

