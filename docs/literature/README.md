# Literature Notes

This page organizes the papers used to frame the project. The important point is not just "what each paper did", but how each one helps explain why this repo focuses on communication-efficient BEV reconstruction.

The downloaded PDFs used while preparing these notes are kept locally under `D:\MEng_Project`. Full paper PDFs are not committed to the public GitHub repo, because the repo should stay lightweight and avoid redistributing papers. The repo keeps the reading notes, links, and experiment-facing takeaways instead.

BibTeX entries for the downloaded papers are in `references.bib`.

## Quick Reading Order

| Order | Paper / Resource | Why It Matters Here |
| --- | --- | --- |
| 1 | V2V4Real dataset and benchmark | The dataset family and benchmark context for this project. |
| 2 | V2X-ViT and CoBEVT | Strong cooperative perception baselines using intermediate feature fusion. |
| 3 | CORE | Closest "cooperative reconstruction" idea, but it reconstructs feature space rather than explicit geometry. |
| 4 | CoCMT | Very strong bandwidth-saving baseline using sparse object queries. |
| 5 | BEV-MAE | Supports the idea that BEV-guided masked reconstruction is meaningful for 3D perception. |

## Project Position In One Sentence

Most strong V2V methods improve detection by fusing hidden feature maps or object queries. This project tests a different route: send a compact multi-height BEV representation and reconstruct the missing physical BEV geometry before downstream perception.

## Same Research Area / Main Baselines

### V2V4Real

- Link: https://mobility-lab.seas.ucla.edu/v2v4real/
- Role in this project: real-world cooperative perception dataset source.
- Why it matters: it gives the real V2V setting, standard splits/benchmarks, and a realistic communication problem.
- How it appears in this repo: the processed data are converted into 8-channel `500 x 500` BEV tensors. The sanity check is in `DATASET_SANITY_CHECK.md`.

The current finished experiments evaluate reconstruction metrics on processed V2V4Real-derived BEV tensors. A later detection stage should report AP@0.5/AP@0.7 on reconstructed/fused outputs.

### V2X-ViT

- Local PDF: `D:\MEng_Project\2203.10638.pdf`
- arXiv: https://arxiv.org/abs/2203.10638
- Main idea: use a Transformer to fuse cooperative perception features across vehicles/infrastructure.
- What it is good at: strong perception accuracy and robustness under noisy multi-agent settings.
- What it costs: it communicates intermediate features, so the transmitted message is much heavier than sparse object queries or compact BEV encodings.
- How to compare against it: use it as a high-accuracy feature-fusion reference. The thesis argument should not be "U-Net is a bigger detector"; it should be "a compact geometric reconstruction can keep useful scene structure with much lower communication cost."

### CoBEVT

- Local PDF: `D:\MEng_Project\2207.02202.pdf`
- arXiv: https://arxiv.org/abs/2207.02202
- Main idea: cooperative BEV perception with sparse Transformer-style fusion.
- What it is good at: strong cooperative BEV representation and perception performance.
- Why it matters here: it is a clean example of the mainstream intermediate-fusion route: learn a powerful feature representation, transmit/fuse it, and optimize directly for perception.
- How to compare against it: CoBEVT-style systems are accuracy-oriented baselines. This project should compare against them on AP when the detector stage is added, but the main advantage should be bandwidth plus interpretable reconstructed geometry.

## Closest Related Ideas

### CORE

- Local PDF: `D:\MEng_Project\2307.11514.pdf`
- arXiv: https://arxiv.org/abs/2307.11514
- Main idea: cooperative reconstruction for multi-agent perception.
- Why it is close: it also uses reconstruction as a way to make collaborative messages more useful.
- Key difference: CORE reconstructs a model feature/observation representation for downstream tasks. This project reconstructs explicit multi-height BEV geometry, which is easier to visualize, inspect, and potentially pass into different downstream perception modules.
- Useful wording for the report: CORE supports the claim that "reconstruction" is a meaningful cooperative perception signal, but this project moves the reconstruction target closer to physical geometry.

### BEV-MAE

- Local PDF: `D:\MEng_Project\2212.05758.pdf`
- arXiv: https://arxiv.org/abs/2212.05758
- Main idea: masked autoencoding for LiDAR point clouds using a BEV-guided strategy.
- Why it matters: it validates the broader idea that masked BEV/point reconstruction can improve 3D understanding.
- Metric connection: BEV-MAE uses Chamfer Distance for masked point reconstruction. That makes Chamfer Distance a natural future metric if this project extends from BEV reconstruction back to point-cloud reconstruction.
- Key difference: BEV-MAE is mainly single-agent self-supervised pre-training. This project is V2V communication-oriented and reconstructs missing/hidden BEV content from another agent.

### CoCMT

- Local PDF: `D:\MEng_Project\2503.13504v1.pdf`
- arXiv: https://arxiv.org/abs/2503.13504
- Main idea: transmit object queries instead of dense feature maps for collaborative perception.
- Important number from the abstract: on V2V4Real, Top-50 object queries use `0.416 Mb` bandwidth and improve AP@70 by `1.1%` over the compared SOTA setting.
- Why it matters: it is the strongest "very low communication" reference among the downloaded papers.
- How this project differs: object-query methods are excellent when the only goal is object detection, but they intentionally discard much of the surrounding geometric context. The multi-height BEV reconstruction route keeps a more complete scene representation, which may matter for road structure, unusual obstacles, and interpretability.

## How The Benchmark Story Should Be Written

### Table 1: Detection Accuracy And Communication

Use this once the downstream detector stage is added.

| Method Type | Example | AP@0.5 | AP@0.7 | Communication |
| --- | --- | ---: | ---: | --- |
| Late fusion | V2V4Real late-fusion baseline | to fill from benchmark | to fill from benchmark | very low, only boxes |
| Intermediate fusion | V2X-ViT / CoBEVT | to fill from benchmark | to fill from benchmark | high, feature maps |
| Object-query sharing | CoCMT | to fill from paper | paper reports strong AP@70 | very low, e.g. `0.416 Mb` |
| Ours | multi-height BEV reconstruction | future detector result | future detector result | compact BEV message |

The expected argument is a trade-off, not a fake claim that every number must be best. If AP is slightly lower than heavy feature-fusion methods but bandwidth and interpretability are much better, that is still a meaningful research result.

### Table 2: Reconstruction Quality

Use this for the part this repo already tests.

| Model | Masked Occ-IoU | Precision | Recall | F1 | Masked RMSE | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| U-Net | see `01_unet_final/results/METRICS_SUMMARY.md` | reported | reported | reported | reported | strongest current baseline |
| Pix2Pix | see `02_pix2pix_final/results/METRICS_SUMMARY.md` | reported | reported | reported | reported | higher precision, lower recall |
| Diffusion v3 | see `03_diffusion_final/results/METRICS_SUMMARY.md` | reported | reported | reported | reported | corrected but over-predicts occupancy |

For a later point-cloud version, add Chamfer Distance beside these BEV metrics.

## Downloaded Paper Manifest

| Local File | Identified Paper | Repo Action |
| --- | --- | --- |
| `2203.10638.pdf` | V2X-ViT: Vehicle-to-Everything Cooperative Perception with Vision Transformer | summarized and linked |
| `2207.02202.pdf` | CoBEVT: Cooperative Bird's Eye View Semantic Segmentation with Sparse Transformers | summarized and linked |
| `2212.05758.pdf` | BEV-MAE: Bird's Eye View Masked Autoencoders for Point Cloud Pre-training | summarized and linked |
| `2307.11514.pdf` | CORE: Cooperative Reconstruction for Multi-Agent Perception | summarized and linked |
| `2503.13504v1.pdf` | CoCMT: Communication-Efficient Cross-Modal Transformer for Collaborative Perception | summarized and linked |
