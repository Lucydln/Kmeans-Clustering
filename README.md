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
