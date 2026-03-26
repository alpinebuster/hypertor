# SafeSurf Darknet 2025
A multi-layer behavioral [dataset](https://drive.google.com/file/d/1UewDNH3eNX4UjfZLtVdrz7N7nRnB8Jz_/view) for darknet traffic detection and classification.

## Getting Started

```sh
/ja4plus/rust/target/release/ja4 --format csv -o "path_to.csv" "path_to.pcap"
```

## 📘 Dataset Description
**SafeSurf Darknet 2025** is a richly labeled network traffic dataset capturing communications across multiple anonymizing technologies like Tor and VPNs.  
Unlike traditional datasets that rely on ports or protocol-based labeling, this dataset is organized by **behavioral context**, enabling advanced research in:

- Darknet behavior classification  
- Encrypted traffic analysis  
- Intrusion Detection / Anomaly Detection Systems (IDS/ADS)

> REF: `https://data.mendeley.com/datasets/kcrnj6z4rm/2`, `https://www.preprints.org/manuscript/202507.1926` and `https://ieeexplore.ieee.org/abstract/document/11073091?casa_token=8kCQyPAxgtMAAAAA:zonB01eqoQ-iPydRXm1B7yjQsoG65w4B_gT2w2jFMhLvM2_zf6Q7cnr_Lk9jA95gcUCz5EtmCIXkXA`

## 🧪 Labeling and Data Collection Methodology
All traffic was **manually generated and labeled** in a controlled environment.  
Sessions were classified based on the **known user activity context**  
(e.g., *watching YouTube over Tor → Video Streaming*).

### Key Characteristics
- Manual labeling only (no heuristic or automated labeling)
- Behavior isolation via dedicated environments
- Time-aligned traffic capture
- High-confidence ground-truth annotations

## 🔐 Privacy-Preserving Technologies Covered
- **Tor**
- **Freenet**
- **I2P**
- **ZeroNet**
- **VPN**

## 🎯 Behavioral Classes Captured
- Browsing  
- Email  
- Chatting  
- Voice over IP (VoIP)  
- File Transfer (FTP)  
- Audio Streaming  
- Video Streaming  
- Peer-to-Peer (P2P) Sharing  
- Normal (non-darknet traffic)

> **Note:** Some behaviors (e.g., VoIP) are not available across all technologies due to service limitations.

## 🧩 Dataset Structure

### 🔹 Layer 1 – Binary Labeling
| Class     | Samples |
|----------|---------|
| Normal   | 360,358 |
| Darknet  | 91,404  |

### 🔹 Layer 2 – Technology-Specific Classification
| Technology | Samples |
|------------|---------|
| Freenet    | 26,284  |
| ZeroNet    | 25,499  |
| I2P        | 22,958  |
| Tor        | 12,546  |
| VPN        | 4,117   |

### 🔹 Layer 3 – Behavioral Labeling
| Behavior          | Samples |
|------------------|---------|
| Browsing         | 33,586  |
| FTP              | 20,214  |
| Video Streaming  | 9,559   |
| P2P Sharing      | 9,392   |
| Email            | 7,873   |
| Audio Streaming  | 5,953   |
| Chatting         | 3,489   |
| VoIP             | 1,338   |

## 📄 Data Format and Features
- **Format:** CSV  
- **Granularity:** One row per network flow  

### Feature Categories
- Timestamps  
- Flow duration  
- Packet and byte counts  
- Inter-arrival time statistics  
- TCP/UDP header metrics  
- Directional and flow-based indicators  

This feature design supports a wide range of machine learning and network security applications.

## 📚 Citation and Licensing
Please cite the dataset in your publications and comply with the licensing terms provided in the dataset repository.

```bib
@article{202507.1926,
	doi = {10.20944/preprints202507.1926.v1},
	url = {https://doi.org/10.20944/preprints202507.1926.v1},
	year = 2025,
	month = {July},
	publisher = {Preprints},
	author = {Qasem Abu Al-Haija and Mohammad J. Obaidat and Ibrahim A. Al-Syouf and Yahea F. Awawdeh and Anas E. Masa’deh},
	title = {SafeSurf Darknet 2025: A Novel Dataset for Darknet Traffic Detection and Analysis},
	journal = {Preprints}
}
```

## 📁 Dataset Access
**Mendeley Data – SafeSurf Darknet 2025**  
**Download Size:** 26.6 MB  

### Files
- `SafeSurf Dataset Layer 1, 2, 3.rar` (26.6 MB)


## 📄 Related Publications
- https://www.preprints.org/manuscript/202507.1926/v1  
- https://ieeexplore.ieee.org/abstract/document/11073091  
