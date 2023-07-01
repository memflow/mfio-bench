#!/bin/sh

pacman -Sy

pacman -S samba --noconfirm

mkdir -p /shares/main
dd if=/dev/urandom of=/shares/main/sample.img bs=1G count=16

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
