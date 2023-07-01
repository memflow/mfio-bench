# mfio network benchmark suite

This repo consists of the following:

- Rust benchmark binary

- `docker-compose` stack

- `Vagrantfile`

You may run `./run_bench.sh vagrant` to run benchmarks automatically, using vagrant backend. If vagrant is not available, you may use docker/podman backend, but note that on linux, smb driver bypasses network interface latency settings.
