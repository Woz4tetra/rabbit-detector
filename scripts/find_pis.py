#!/usr/bin/env python3
"""Scan the local network for Raspberry Pi devices by MAC address OUI."""

import argparse
import ipaddress
import re
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Raspberry Pi Foundation / Raspberry Pi Trading OUI prefixes
PI_OUIS = {
    "b8:27:eb",  # Raspberry Pi Foundation (Pi 1, 2, 3, Zero W)
    "dc:a6:32",  # Raspberry Pi Trading (Pi 4, Pi Zero 2W)
    "e4:5f:01",  # Raspberry Pi Trading (Pi 4 rev 1.4+, Pi 400)
    "28:cd:c1",  # Raspberry Pi Trading (Pi 5)
    "d8:3a:dd",  # Raspberry Pi Trading (Pi 5)
}


def get_subnet_for_interface(iface: str) -> str:
    """Return the subnet in CIDR notation for a given network interface."""
    result = subprocess.run(["ip", "addr", "show", iface], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: interface '{iface}' not found.", file=sys.stderr)
        sys.exit(1)
    for line in result.stdout.splitlines():
        line = line.strip()
        if line.startswith("inet "):
            cidr = line.split()[1]  # e.g. 192.168.1.5/24
            network = ipaddress.IPv4Network(cidr, strict=False)
            return str(network)
    print(f"Error: no IPv4 address found on interface '{iface}'.", file=sys.stderr)
    sys.exit(1)


def ping(ip: str) -> None:
    subprocess.run(
        ["ping", "-c", "1", "-W", "1", ip],
        capture_output=True,
    )


def read_arp_table(iface: str) -> dict[str, str]:
    """Return {ip: mac} from the kernel ARP table, filtered to iface."""
    result = subprocess.run(["arp", "-n", "-i", iface], capture_output=True, text=True)
    entries: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        # arp -n columns: Address HWtype HWaddress Flags Iface
        if len(parts) >= 3 and re.match(r"([0-9a-f]{2}:){5}[0-9a-f]{2}", parts[2], re.I):
            entries[parts[0]] = parts[2].lower()
    return entries


def hostname(ip: str) -> str:
    try:
        return socket.gethostbyaddr(ip)[0]
    except socket.herror:
        return ""


def is_pi_mac(mac: str) -> bool:
    return mac[:8].lower() in PI_OUIS


def main() -> None:
    parser = argparse.ArgumentParser(description="Find Raspberry Pis on the local network.")
    parser.add_argument("interface", help="Network interface to scan (e.g. eth0, wlan0)")
    args = parser.parse_args()

    subnet = get_subnet_for_interface(args.interface)
    network = ipaddress.IPv4Network(subnet, strict=False)
    hosts = [str(h) for h in network.hosts()]

    print(f"Scanning {subnet} on {args.interface} ({len(hosts)} hosts) ...", flush=True)

    with ThreadPoolExecutor(max_workers=64) as pool:
        futures = {pool.submit(ping, ip): ip for ip in hosts}
        for f in as_completed(futures):
            f.result()  # consume so exceptions surface

    arp = read_arp_table(args.interface)
    pis = [(ip, mac) for ip, mac in arp.items() if is_pi_mac(mac)]

    if not pis:
        print("No Raspberry Pis found.")
        sys.exit(0)

    print(f"\nFound {len(pis)} Raspberry Pi(s):\n")
    print(f"  {'IP':<18} {'MAC':<19} {'Hostname'}")
    print(f"  {'-'*17} {'-'*17} {'-'*30}")
    for ip, mac in sorted(pis, key=lambda x: ipaddress.IPv4Address(x[0])):
        name = hostname(ip)
        print(f"  {ip:<18} {mac:<19} {name}")


if __name__ == "__main__":
    main()
