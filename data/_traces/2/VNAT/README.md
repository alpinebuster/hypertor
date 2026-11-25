# VNAT

This dataset [vnat](https://www.ll.mit.edu/r-d/datasets/vpnnonvpn-network-application-traffic-dataset-vnat) is a collection of labelled PCAP files, both encrypted and unencrypted, across 10 applications. It was created to assist the development of machine learning tools that would allow operators to see the traffic categories of both encrypted and unencrypted traffic flows. In particular, features of the network packet traffic timing and size information (both inside of and outside of the VPN) can be leveraged to predict the application category that generated the traffic.

## Traffic Details

The dataset consists of 36.1 GB, 33,711 connections, and approximately 272 hours of packet capture from five traffic categories, as shown in the table below:

*Table 1: Traffic Details*

| Traffic Category       | Applications            | Filename Keywords       |
| ---------------------- | ----------------------- | ----------------------- |
| Streaming              | Vimeo, Netflix, Youtube | vimeo, netflix, youtube |
| VoIP                   | Zoiper                  | Voip                    |
| Chat                   | Skype                   | skype-chat              |
| Command & Control (C2) | SSH, RDP                | ssh, rdp                |
| File Transfer          | SFTP, RSYNC, SCP        | sftp, rsync, scp        |

To produce the dataset, virtual subnetworks for each traffic category were created. Each subnetwork contains a client, a client DNS server, a VPN client, and a VPN server. The Skype subnetwork contains an additional client to allow for bidirectional chat. The video streaming and web browsing subnetworks were connected to the Internet to enable access to Firefox, Chrome, YouTube, Netflix, and Vimeo. VPN traffic was captured between the VPN client and the VPN server. Separately, non-VPN traffic is captured between the VPN client and the application layer.

Netflix, YouTube, Zoiper, and Vimeo network traffic were generated manually. However, the File Transfer network traffic was generated with the assistance of randomized scripts. The Chat category was created by playing back chat messages available on <https://github.com/freeCodeCamp/gitter-history>. For the C2 category, the RDP traffic was manually generated, whereas the ssh traffic was created with randomized scripts that executed shell commands. All traffic was captured using tcpdump and outputted in the libpcap compatible PCAP format.

![traffic collection](assets/pcap_vnat.png)

## File Naming Format

The data was captured using TCP dump on an isolated subnetwork, where only network traffic from the desired application was present. Since all applications captured encrypt the packet payloads, no obfuscation of the payload is required. Since the packets were captured on an isolated subnet created for only this purpose, no obfuscation of packet header data was required. After capturing the PCAP data, files were labeled according to the application run during the capture using the following format:

*Table 2: File Naming Format*

| Capture Type | File Naming Format                        |
| ------------ | ----------------------------------------- |
| VPN          | vpn_<filename keyword>_capture<#>.pcap    |
| NON-VPN      | nonvpn_<filename keyword>_capture<#>.pcap |

More information about the dataset can be found in our related publication, available at <https://ieeexplore.ieee.org/abstract/document/10044382>.
