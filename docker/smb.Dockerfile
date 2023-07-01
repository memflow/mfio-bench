FROM ghcr.io/servercontainers/samba:smbd-only-latest

RUN apk add iproute2

#ENTRYPOINT "runsvdir -P /container/config/runit"
