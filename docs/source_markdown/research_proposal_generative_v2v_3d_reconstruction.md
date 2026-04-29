# **Project Proposal: Generative 3D Point Cloud Reconstruction via Multi-Height BEV for V2V Cooperative Perception**

## **1\. Problem Statement & Motivation**

In Vehicle-to-Vehicle (V2V) cooperative perception, sharing raw 3D LiDAR point clouds yields the highest perception accuracy but requires impractical communication bandwidth. Current mainstream methods transmit heavily compressed intermediate feature maps or simply 3D bounding boxes. However, this often discards critical geometric context and structural details necessary for downstream planning and robustness.

This project proposes a novel paradigm: encoding 3D point clouds into a **Multi-Height Bird's Eye View (BEV)** representation for ultra-low bandwidth transmission, and deploying a lightweight generative model at the receiver to reconstruct the 3D point cloud. This reconstructed point cloud is then fed into standard 3D object detectors to maintain high perception accuracy.

## **2\. Proposed Technical Pipeline**

* **Sender (Vehicle A):** Captures 3D LiDAR point clouds and slices them along the Z-axis to create a Multi-Height BEV representation. This effectively preserves vertical structures while compressing the data into a 2D format suitable for transmission.  
* **Transmission:** The compressed Multi-Height BEV is transmitted via the V2V channel.  
* **Receiver (Vehicle B):** Utilizes a lightweight generative model (U-Net) to decode the received BEV and reconstruct the missing 3D point cloud geometry.  
* **Perception Task:** The reconstructed 3D point cloud is processed by a pre-trained 3D object detector (e.g., PointPillars) to evaluate the final Average Precision (AP).

## **3\. Strategic Decisions & Rationale**

Based on our preliminary experiments and literature review, we have made the following strategic design choices to ensure the project's practical viability:

* **U-Net over Diffusion Models:** While Diffusion Models (e.g., DDPM) offer state-of-the-art visual fidelity in generative tasks, their iterative denoising process introduces hundreds of milliseconds of latency, making them fundamentally incompatible with the strict real-time constraints (\< 100ms) of autonomous driving. Our experiments confirm that a well-designed U-Net offers the optimal trade-off between inference speed, reconstruction quality, and bandwidth efficiency.  
* **Optimizing Multi-Height BEV:** A core contribution will be optimizing the BEV encoding. We will ablate the number of height slices (e.g., 4 vs. 8 layers) and the encoded features (e.g., max-Z, density, intensity) within each slice to maximize the information available for the U-Net decoder.  
* **Perceptual Loss (Task-Aligned Loss):** Instead of training the U-Net using only geometric losses (e.g., Chamfer Distance) which force the model to reconstruct irrelevant background details (like trees or ground points), we will incorporate a Perceptual Loss. By utilizing the feature space of a pre-trained 3D detector, the U-Net is penalized for failing to reconstruct features crucial for object detection (e.g., vehicle contours and pedestrian shapes).

## **4\. Experimental Plan & Benchmarks**

The project will utilize the **V2V4Real** dataset, a large-scale real-world V2V perception dataset collected by the UCLA Mobility Lab.

| Evaluation Dimension | Metrics | Target Baseline / Competitor |
| :---- | :---- | :---- |
| Perception Accuracy | AP@0.5, AP@0.7 | Aim to approach intermediate-fusion SOTAs like **CoBEVT (AP@0.5: 66.5)** or **V2X-ViT**. |
| Communication Efficiency | Transmission Bitrate (MB/frame) | Aim for bandwidth significantly lower than standard feature sharing (e.g., \~0.20 MB) and approaching extreme compression methods like **CoCMT**. |
| Reconstruction Quality | Chamfer Distance (CD) | Measure the geometric fidelity of the reconstructed point cloud vs. the raw LiDAR data. |

## **5\. Relevant Literature & References**

The following papers provide the academic foundation and competitive baselines for our approach. They are categorized to highlight the gap our research addresses.

### **Category A: V2V Perception & The V2V4Real Ecosystem**

* **1\. V2V4Real: A Real-World Large-Scale Dataset for Vehicle-to-Vehicle Cooperative Perception (CVPR 2023\)**  
  *Relevance:* The core dataset for this project, developed by Jiaqi Ma's team at UCLA Mobility Lab. It provides the real-world LiDAR data, bounding boxes, and standard benchmarks (No Fusion, Late Fusion, V2VNet, V2X-ViT, CoBEVT) required to validate our method. [\[Dataset Link\]](https://mobility-lab.seas.ucla.edu/v2v4real/)  
* **2\. V2X-ViT: Vehicle-to-Everything Cooperative Perception with Vision Transformer (ECCV 2022\)**  
  *Relevance:* A primary baseline. It uses a unified Transformer architecture to fuse BEV features from multiple agents. We must demonstrate that our generative reconstruction approach can achieve competitive accuracy against this strong intermediate fusion method while saving bandwidth.  
* **3\. CoBEVT: Cooperative Bird's Eye View Semantic Segmentation with Sparse Transformers (CoRL 2022\)**  
  *Relevance:* Currently holds the state-of-the-art perception accuracy (AP@0.5 \~66.5) on the V2V4Real benchmark.  
* **4\. Recent Dataset Extensions: V2U4Real and V2X-Radar (2024/2026)**  
  *Relevance:* Recent papers extending the V2V4Real paradigm to include UAVs ([V2U4Real](https://arxiv.org/html/2603.25275v1)) and 4D Radar ([V2X-Radar](https://openreview.net/forum?id=sTKsFIVqik&referrer=%5Bthe+profile+of+Jiaqi+Ma%5D\(/profile?id%3D~Jiaqi_Ma4\))) by the same core research community, demonstrating the active and evolving nature of this dataset ecosystem.

### **Category B: Generative Reconstruction & BEV Representations**

* **5\. CORE: Cooperative Reconstruction for Multi-Agent Perception (ICCV 2023\)**  
  *Relevance:* The most conceptually similar V2V paper. CORE introduces the idea of using reconstruction as a supervisory task to improve collaborative messages. However, it only reconstructs BEV features, whereas our project explicitly rebuilds the 3D point cloud geometry.  
* **6\. BEV-MAE: Bird's Eye View Masked Autoencoders for Point Cloud Pre-training (AAAI 2024\)**  
  *Relevance:* Proves that autoencoders can successfully reconstruct masked 3D points guided by BEV representations to improve downstream 3D detection. This validates the feasibility of our "BEV to Point Cloud" reconstruction pipeline, albeit BEV-MAE is for single-vehicle pre-training, not V2V communication.  
* **7\. CoCMT: Communication-Efficient Cooperative Perception via Object Queries (2024/2025)**  
  *Relevance:* The extreme baseline for bandwidth efficiency. It transmits only object queries rather than dense features. We will contrast our method against CoCMT, arguing that while they achieve extreme compression, our generative approach retains vital geometric context (drivable areas, unclassified obstacles) that object queries discard.