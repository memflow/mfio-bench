#!/bin/sh

prev=0

socat - TCP-LISTEN:12345,fork,reuseaddr | \
	while read line; do
		echo "${line}ms"
		if [ "$prev" -ne "0" ]; then
			echo tc qdisc del dev eth0 root netem | docker exec -i docker-smb-1 /bin/sh
		fi
		if [ "$line" -ne "0" ]; then
			echo tc qdisc add dev eth0 root netem delay "${line}ms" | docker exec -i docker-smb-1 /bin/sh
		fi
		prev="$line"
	done
