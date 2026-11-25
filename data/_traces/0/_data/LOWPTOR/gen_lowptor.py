#!/usr/bin/env python3

import os
import argparse
import subprocess
import sys
import time
import signal
from pathlib import Path
from datetime import datetime
from threading import Thread, Event


####################################################
# Utilities
####################################################


def run(cmd: list[str], check=True, shell=False, silent=True):
    print(f"[CMD] {cmd}")
    if silent:
        return subprocess.run(cmd, check=check, shell=shell,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL,)
    else:
        return subprocess.run(cmd, check=check, shell=shell)

def which(cmd: str):
    return subprocess.run(
        ["which", cmd],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    ).returncode == 0

def normalize_path(p: str, base: Path) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = (base / path).resolve()
    return path

def find_pcaps(dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for d in dirs:
        if not d.is_dir():
            print(f"[WARN] {d} is not a directory")
            continue
        files.extend(d.rglob("*.pcap"))
    return files

def is_root() -> bool:
    return os.geteuid() == 0


####################################################
# Argument parsing
####################################################


if not is_root():
    print("Please run as root: sudo python3 gen_lowptor.py")
    sys.exit(1)

parser = argparse.ArgumentParser()
parser.add_argument("-d", default="cicdarknet", help="TOR source dataset")
parser.add_argument("-n", default="EWTD", help="NONTOR source dataset")
parser.add_argument("-r", type=float, default=0.001, help="TOR ratio")
parser.add_argument("-m", type=float, default=1000, help="Max Mbps")
parser.add_argument("-i", default="", help="TOR PCAP dir")
parser.add_argument("-o", default="LOWPTOR", help="Output base dir")
args = parser.parse_args()

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_BASE_DIR = normalize_path(args.o, SCRIPT_DIR)


####################################################
# Logging
####################################################


class Tee:
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

logfile = f"gen_lowptor-{datetime.now().strftime('%F_%H-%M-%S')}.log"
log_f = open(logfile, "a", buffering=1)

# Keep the original stdout
orig_stdout = sys.__stdout__
orig_stderr = sys.__stderr__

sys.stdout = Tee(orig_stdout, log_f)
sys.stderr = Tee(orig_stderr, log_f)

print(f"[INFO] Log file: {logfile}")


####################################################
# Check dependencies
####################################################


INSTALL_SCRIPT = SCRIPT_DIR / "install_deps.sh"
missing_tools = [tool for tool in ["tcpreplay", "tcpdump", "ip"] if not which(tool)]
if missing_tools:
    print(f"[WARN] Missing tools: {missing_tools}")

    if not INSTALL_SCRIPT.exists():
        print(f"[ERROR] install_deps.sh not found at {INSTALL_SCRIPT}")
        sys.exit(1)

    print("[INFO] Running install_deps.sh to install dependencies...")
    subprocess.run(
        ["sudo", "bash", str(INSTALL_SCRIPT)],
        check=True,
    )

    # re-check after installation
    still_missing = [tool for tool in missing_tools if not which(tool)]
    if still_missing:
        print(f"[ERROR] Tools still missing after install: {still_missing}")
        sys.exit(1)

print("[INFO] All dependencies are installed")


####################################################
# Dataset paths
####################################################


if args.d == "cicdarknet":
    TOR_PCAP_DIR = SCRIPT_DIR / "ISCX-Tor-NonTor-2017" / "Tor"
elif args.d == "tcub":
    TOR_PCAP_DIR = SCRIPT_DIR / "darknet-dataset-2020" / "tor"
elif args.d == "safesurf2025":
    TOR_PCAP_DIR = SCRIPT_DIR / "SafeSurf-Darknet-2025" / "Tor"
else:
    raise RuntimeError("Unknown dataset and TOR_PCAP_DIR not provided")

if args.i:
    TOR_PCAP_DIR = normalize_path(args.i, SCRIPT_DIR)

MAWI_PCAP_DIR = SCRIPT_DIR / "MAWI"
VNAT_PCAP_DIR = SCRIPT_DIR / "VNAT"
EWTD_PCAP_DIR = SCRIPT_DIR / "EWTD"

NON_TOR_DIRS = [VNAT_PCAP_DIR]
if args.n.upper() == "VNAT":
    NON_TOR_DIRS = [VNAT_PCAP_DIR]
elif args.n.upper() == "MAWI":
    NON_TOR_DIRS = [MAWI_PCAP_DIR]
elif args.n.upper() == "EWTD":
    NON_TOR_DIRS = [EWTD_PCAP_DIR]
else:
    raise ValueError(f"Unsupported dataset: {args.n}")

NON_TOR_FILES = find_pcaps(NON_TOR_DIRS)
print(f"[INFO] Found {len(NON_TOR_FILES)} NON-TOR pcaps")
for f in NON_TOR_FILES:
    print(f"  {f}")


####################################################
# Bandwidth
####################################################


TOR_MBPS = args.m * args.r
OTHER_MBPS = args.m * (1 - args.r)

print(f"[INFO] TOR_MBPS={TOR_MBPS:.4f}, OTHER_MBPS={OTHER_MBPS:.4f}")


####################################################
# veth setup
####################################################


def ensure_veth():
    if subprocess.run(
        ["ip", "link", "show", "veth_tcpreplay"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode != 0:
        run(["sudo", "ip", "link", "add", "veth_tcpreplay",
             "type", "veth", "peer", "name", "veth_tcpdump"], silent=False)

    run(["sudo", "ip", "link", "set", "veth_tcpreplay", "up"], silent=False)
    run(["sudo", "ip", "link", "set", "veth_tcpdump", "up"], silent=False)
    run(["sudo", "ip", "link", "set", "dev", "veth_tcpreplay", "mtu", "9216"], silent=False)
    run(["sudo", "ip", "link", "set", "dev", "veth_tcpdump", "mtu", "9216"], silent=False)

def cleanup_veth():
    # As long as one end is deleted, the other end will disappear automatically
    ret = subprocess.run(
        ["ip", "link", "show", "veth_tcpreplay"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode

    if ret == 0:
        print("[CLEANUP] deleting veth pair (veth_tcpreplay <-> veth_tcpdump)")
        run(["sudo", "ip", "link", "del", "veth_tcpreplay"], silent=False)
    else:
        print("[CLEANUP] veth not present, nothing to delete")

ensure_veth()


####################################################
# Main loop
####################################################


TOR_CLASS_PATTERNS = {
    "browsing": ["browsing_*.pcap"],
    "chat": ["chat_*.pcap"],
    "email": ["mail_*.pcap"],
    "streaming": [
        "audio_*.pcap", "tor_spotify*.pcap",
        "video_*.pcap", "tor_vimeo*.pcap", "tor_youtube*.pcap"
    ],
    "ftp": ["file-transfer_*.pcap"],
    "voip": ["voip_*.pcap"],
    "p2p": ["p2p_*.pcap", "tor_p2p*.pcap"],
}
if args.d == "tcub":
    TOR_CLASS_PATTERNS = {
        "browsing": ["tor-browsing*.pcap"],
        "chat": ["tor-chat_*.pcap"],
        "email": ["tor-email_*.pcap"],
        "streaming": [
            "tor-audio_*.pcap",
            "tor-video_*.pcap",
        ],
        "ftp": ["tor-ftp*.pcap"],
        "voip": ["tor-voip_*.pcap"],
        "p2p": ["tor-p2p_*.pcap"],
    }

if args.d == "safesurf2025":
    TOR_CLASS_PATTERNS = {
        "browsing": [
            "tor-browsing*.pcap",
            "*browsing*.pcap",
        ],
        "chat": ["tor-chat_*.pcap", "*chat*.pcap"],
        "email": ["tor-email_*.pcap", "*email*.pcap"],
        "streaming": [
            "tor-audio_*.pcap",
            "tor-video_*.pcap",
            "*Audio*.pcap",
            "*video*.pcap",
        ],
        "ftp": ["tor-ftp*.pcap", "*ftp*.pcap"],
        "voip": ["tor-voip_*.pcap", "*VOIP*.pcap", "*voip*.pcap"],
        "p2p": ["tor-p2p_*.pcap"],
    }


def replay_pcaps(pcaps: list[Path], mbps: float, stop_event: Event, name: str):
    """Serially play the given pcaps; When a stop_event is encountered, the currently running tcpreplay will be stopped as soon as possible and returned.
    """
    procs: list[subprocess.Popen] = []
    try:
        for p in pcaps:
            if stop_event.is_set():
                print(f"[{name}] stop_event is set before launching {p}, exiting")
                break

            cmd = ["tcpreplay", "-i", "veth_tcpreplay", f"--mbps={mbps}", str(p)]
            print(f"[{name}] launching: {cmd}")
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            procs.append(proc)

            # Wait for the current tcpreplay to end, or terminate it when stop_event is triggered
            while True:
                ret = proc.poll()
                if ret is not None:
                    # `tcpreplay` ended
                    if ret != 0:
                        print(f"[{name}] tcpreplay returned {ret} for {p}, setting stop_event")
                        stop_event.set() # Notify the other end to stop
                    else:
                        print(f"[{name}] finished {p}")
                    break

                # If other threads request to stop, immediately terminate this tcpreplay
                if stop_event.is_set():
                    print(f"[{name}] stop_event set, terminating tcpreplay pid={proc.pid}")
                    try:
                        proc.terminate()
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    break

                # After a brief sleep, continue the poll
                time.sleep(0.2)

        # If you have completed your own pcaps (naturally ended) normally, you will also notify the other end to stop
        print(f"[{name}] finished all pcaps, setting stop_event to notify peer")
        stop_event.set()
    finally:
        # Clear any possible remaining child processes within this function
        for p in procs:
            if p.poll() is None:
                try:
                    p.terminate()
                    p.wait(timeout=1)
                except Exception:
                    p.kill()
                    p.wait()

for cls, patterns in TOR_CLASS_PATTERNS.items():
    tor_pcaps: list[Path] = []
    for pat in patterns:
        tor_pcaps.extend(TOR_PCAP_DIR.rglob(pat))

    if not tor_pcaps:
        print(f"[WARN] No TOR pcaps for class {cls}")
        continue
    print(f"[INFO] Found {len(tor_pcaps)} TOR pcaps for class '{cls}':")
    for f in tor_pcaps:
        print(f"  {f}")

    out_dir = OUTPUT_BASE_DIR / args.d / cls
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.n.lower()
    output_file = out_dir / f"{prefix}-lowptor-{args.d}-{cls}-{args.r}.pcap"

    print(f"[INFO] CLASS={cls} → {output_file}")
    # If the `output_file` already exists, it will be overwritten unconditionally
    tcpdump = subprocess.Popen(
        ["sudo", "tcpdump", "-i", "veth_tcpdump", "-w", str(output_file)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setsid,
    )

    time.sleep(6)

    stop_event = Event()
    t_tor = Thread(target=replay_pcaps, args=(tor_pcaps, TOR_MBPS, stop_event, f"TOR-{cls}"))
    t_non = Thread(target=replay_pcaps, args=(NON_TOR_FILES, OTHER_MBPS, stop_event, f"NON-{cls}"))

    t_tor.start()
    t_non.start()

    while True:
        if not t_tor.is_alive() or not t_non.is_alive():
            print(f"[INFO] detected a thread finished for class {cls}, signalling stop_event")
            stop_event.set()
            break
        # Sleep a little to avoid busy-wait
        time.sleep(0.5)

    t_tor.join()
    t_non.join()

    print(f"[INFO] stopping tcpdump for class {cls}")
    os.killpg(os.getpgid(tcpdump.pid), signal.SIGINT)
    tcpdump.wait()
    print(f"[INFO] stopped for class {cls}")
    time.sleep(2)


####################################################
# Cleanup
####################################################


print("[CLEANUP] killing leftover processes")
subprocess.run(["sudo", "pkill", "-TERM", "tcpreplay"], check=False)
subprocess.run(["sudo", "pkill", "-INT", "tcpdump"], check=False)

cleanup_veth()
print("[DONE] LOWPTOR traffic replay completed!")
