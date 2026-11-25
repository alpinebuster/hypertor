# Five-week DoH Dataset collected on ISP backbone lines

This dataset is an additional DoH dataset used for researching DoH traffic data drift phenomena. It contains anonymized packet captures (pcaps) from the following days:

* 2022-11-28  
* 2022-12-05 
* 2022-12-12  
* 2022-12-19
* 2022-12-26

The traffic was captured on the CESNET2 network and anonymized. The packet capturing and anonymization follow the methodology described in [1]. The list of IP addresses used for DoH recognition is also included within the dataset in doh\_resolver\_ip.csv file.
The structure of the dataset is as follows:


```
.
├── doh_resolver_ip.csv
├── pcap
│   ├── 2022-11-28
│   │   ├── DoH-20221128180002.pcapng
│   │   └── HTTPS-20221128180002.pcapng
│   ├── 2022-12-05
│   │   ├── DoH-20221205180001.pcapng
│   │   └── HTTPS-20221205180001.pcapng
│   ├── 2022-12-12
│   │   ├── DoH-20221212180001.pcapng
│   │   └── HTTPS-20221212180001.pcapng
│   ├── 2022-12-19
│   │   ├── DoH-20221219180001.pcapng
│   │   └── HTTPS-20221219180001.pcapng
│   └── 2022-12-26
│       ├── DoH-20221226180001.pcapng
│       └── HTTPS-20221226180001.pcapng
└── README.md
```

[1] Jeřábek, K., Hynek, K., Čejka, T., & Ryšavý, O. (2022). Collection of datasets with DNS over HTTPS traffic. Data in Brief, 42, 108310. [LINK](https://www.sciencedirect.com/science/article/pii/S2352340922005121)
