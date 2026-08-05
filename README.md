# Ocular Verification for Virtual Reality

Virtual reality (VR) headsets (e.g., Meta Quest, Apple Vision Pro) provide a seamless user experience due to their fast, frictionless
interaction with the physical world in a simulated environment. User authentication relies on biometric cues such as iris in such headsets. However, traditional iris recognition protocols may not be adequate in cases of unconstrained acquisition, which is typical of VR-based data. In this work, we examine three crucial aspects: (1) evaluating ISO/IEC 29794-6 iris quality metrics on VRBiom dataset and analyzing their limitations, (2) addressing data-specific challenges such as off-axis gaze, non-uniform illumination, and specular reflection using generative models, and (3) performing unimodal (iris, periocular) recognition and multimodal score-level fusion (iris + periocular). We observe that some metrics (e.g., margin adequacy) fail on VR-acquired data; whereas, image adjustments primarily benefit periocular recognition, and multimodal fusion lowers EER by ∼11% over unimodal iris recognition performance.

## SETUP

## Specularity Removal (UnReflect)

We used the UnReflectAnything framework, an RGB-only single-image highlight removal pipeline.

### Installation

```bash
pip install unreflectanything==1.1.1
ura download --weights
```

### Run

```bash
python UnReflect/unreflect.py --in <input_frames_path> --out <output_frames_path>
```



## Non-Uniform Illumination Restoration (UNIR-Net)

We used the UNIR-Net framework, an illumination restoration network that combines illumination enhancement, attention mechanisms, visual refinement, and contrast correction to improve image quality and visibility.

### Installation

```bash
git clone https://github.com/xingyumex/UNIR-Net.git
cp unirnet.py UNIR-Net/   # pretrained weights are included in the cloned repo
```

### Run

```bash
cd UNIR-Net
python inference_unirnet.py --input <input_frames_path> --output <output_frames_path>
```

## Acknowledgment
This material is based upon work supported by the National Science Foundation under Grant No. 2349771 through the HUMANS MOVE REU Site at the University of Wyoming.
