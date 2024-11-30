# KMeans Clustering and Pattern Analysis for IPv4 Addresses

This repository provides tools to analyze IPv4 addresses using K-Means clustering, Principal Component Analysis (PCA), and subnet calculation techniques. It enables network administrators and data analysts to uncover patterns, detect anomalies, and group similar IPs based on usage or other criteria.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Scripts Overview](#scripts-overview)
  - [Running K-Means Clustering](#running-k-means-clustering)
  - [Subnet Calculation](#subnet-calculation)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Introduction

IPv4 address analysis is crucial for network monitoring, anomaly detection, and resource allocation. This repository provides scripts for clustering IPv4 addresses into meaningful groups and calculating subnets for IP ranges. The project uses machine learning techniques and network-specific algorithms to extract insights.

## Features

- **IPv4 Clustering**: Convert IPv4 addresses into numerical format and apply K-Means clustering.
- **PCA for Visualization**: Reduce dimensionality for visualizing clusters in 2D space.
- **Elbow Method**: Determine the optimal number of clusters using distortion and inertia metrics.
- **Subnet Calculation**: Identify inclusive subnets for any two IPv4 addresses.
- **Custom IP Analysis**: Focus on specific accounts or time ranges for targeted analysis.
- **Outlier Detection**: Exclude outliers based on distance from cluster centroids and usage thresholds.

## Requirements

- Python 3.6+
- Libraries:
  - Pandas
  - NumPy
  - Matplotlib
  - Scikit-learn
  - ipaddress

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Lucydln/Kmeans-Clustering.git
   cd Kmeans-Clustering
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Usage

### Scripts Overview

- **`ip_kmeans_new.py`**: Main script for applying K-Means clustering to IPv4 addresses and visualizing results.
- **`IP_pattern.py`**: Contains utilities for subnet calculation and IP range analysis.
- **`test.py`**: Example usage of the clustering script.

### Running K-Means Clustering

1. Modify the `df_file` variable in `ip_kmeans_new.py` to point to your dataset (e.g., `ip_for_top_acct.csv`).
2. Run the script:
   ```bash
   python ip_kmeans_new.py
   ```

## Follow the Prompts

Follow the prompts to determine the number of clusters using the Elbow Method or specify a value directly.

### View Output:

- Cluster assignments.
- Scatter plots of PCA-reduced dimensions with annotated IPs and cluster centers.

## Subnet Calculation

Use the `IP_pattern.py` script to calculate an inclusive subnet for two IP addresses:
```python
from IP_pattern import calc_inclusive_subnet

subnet = calc_inclusive_subnet('192.168.1.1', '192.168.1.100')
print(f"The inclusive subnet is: {subnet}")
```

## Examples

### Example Input

**CSV File (`ip_for_top_acct.csv`):**
```csv
account_id,xrealip
yvC1QJC9TjyqSUwLMIdpYA,192.168.1.1
yvC1QJC9TjyqSUwLMIdpYA,192.168.1.2
yvC1QJC9TjyqSUwLMIdpYA,192.168.1.3
```
### Example Clustering Visualization

- A scatter plot with clusters labeled and IPs annotated in PCA-reduced dimensions.
- Cluster centers highlighted for easy identification.

### Example Subnet Output

'''bash
Inclusive subnet for IPs 192.168.1.1 and 192.168.1.100 is: 192.168.1.0/25
'''

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch:
   '''bash
   git checkout -b feature-name
   '''
3. Commit your changes:
   '''bash
   git commit -m "Add feature-name"
   '''
4. Push to your branch:
   '''bash
   git push origin feature-name
   '''
5. Open a pull request.

## License

This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.



   
