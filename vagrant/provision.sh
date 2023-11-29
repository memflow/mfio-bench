#!/bin/sh

pacman -Sy

pacman -S samba --noconfirm

mkdir -p /shares/main
dd if=/dev/urandom of=/shares/main/sample.img bs=1G count=8

cat > /etc/samba/smb.conf <<EOF
# Global parameters
[global]
Server role: ROLE_STANDALONE

        dns proxy = No
        load printers = No
        log file = /dev/stdout
        map to guest = Bad User
        obey pam restrictions = Yes
        passdb backend = smbpasswd
        printcap name = /dev/null
        security = USER
        server role = standalone server
        server string = Samba Server
        smb1 unix extensions = No
        fruit:aapl = yes
        fruit:model = TimeCapsule
        idmap config * : backend = tdb
        acl allow execute always = Yes
        wide links = Yes


[main]
        guest ok = Yes
        path = /shares/main
        read only = No
EOF

systemctl enable smb --now

# build mfio-netfs server
pacman -S rustup git --noconfirm
rustup toolchain add stable
git clone https://github.com/memflow/mfio
cd mfio
cargo build --release -p mfio-netfs --all-features --bin mfio-netfs-server
cp target/release/mfio-netfs-server /usr/local/bin/
