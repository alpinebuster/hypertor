#!/bin/bash

echo "========== Operating System Information =========="
cat /etc/os-release
echo "Kernel Version: $(uname -r)"
echo "System Architecture: $(uname -m)"
echo "Hostname: $(hostname)"

echo
echo "========== CPU Information =========="
lscpu

echo
echo "========== Memory Information =========="
free -h

echo
echo "========== Network Interface Information =========="
ip -br addr
echo

for i in /sys/class/net/*; do
    iface=$(basename "$i")
    echo "Interface Name: $iface"
    echo "MAC Address: $(cat "$i/address" 2>/dev/null)"
    echo "Network Speed: $(cat "$i/speed" 2>/dev/null) Mbps"
    echo
done

echo
echo "========== Disk Information =========="
lsblk
echo
df -h

echo
echo "========== System Uptime =========="
uptime
