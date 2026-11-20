# MAWI
The MAWI dataset captures only the first 96 bytes of each packet (snaplen = 96). As a result, the complete TLS handshake is truncated and cannot be fully observed, making it impossible to extract JA4 (TLS) fingerprints.

## Getting Started
Note:
 * -k: Retain `*.gz` files
 * -P 8:8 parallel (adjust according to CPU)

```sh
ls *.pcap.gz | xargs -n 1 -P 8 gunzip -k
```
