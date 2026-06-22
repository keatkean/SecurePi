# Available IMX500 AI Models (`.rpk`)

When you install the `imx500-all` package on a Raspberry Pi using `sudo apt install imx500-all`, it automatically downloads a suite of pre-compiled neural network models to the `/usr/share/imx500-models/` directory.

The SecurePi application is designed to work with **Object Detection** models. However, the IMX500 camera supports several different types of AI tasks. Here is a summary of the common models included in that directory and what they do.

> [!IMPORTANT]
> **Supported Models for SecurePi**
> SecurePi requires **Object Detection** models that output bounding boxes. If you pass a Pose Estimation, Image Classification, or Semantic Segmentation model to SecurePi, the script will crash.
> 
> **Safe to use without crashing:**
> - `imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk`
> - `imx500_network_nanodet_plus_416x416_pp.rpk`

---

## 📦 1. Object Detection Models
These models draw "bounding boxes" around objects they recognize. **These are the only models that work with SecurePi.**

- **`imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk`**
  - **Function:** This is the **default model** used by SecurePi. It provides a great balance of speed and accuracy. 
  - **Detects:** 80 COCO categories (people, bags, animals, vehicles, etc.).

- **`imx500_network_nanodet_plus_416x416_pp.rpk`**
  - **Function:** An alternative object detection model based on NanoDet Plus. It uses a different post-processing pipeline but is fully supported by SecurePi. It is often faster and better at detecting smaller objects.
  - **Detects:** 80 COCO categories.
  - **Usage in SecurePi:** `python securePi.py --model /usr/share/imx500-models/imx500_network_nanodet_plus_416x416_pp.rpk`

---

## 🦴 2. Pose Estimation Models
These models do not draw boxes. Instead, they locate human joints (eyes, nose, shoulders, elbows, knees) to map out a human skeleton. *(Will crash if passed to SecurePi)*.

- **`imx500_network_posenet_mobilenet_v1_100_257x257_ptq_pp.rpk`**
  - **Function:** PoseNet model for tracking human movement and posture.

- **`imx500_network_movenet_single_pose_lightning_192x192_ptq_pp.rpk`**
  - **Function:** MoveNet Lightning. An extremely fast model designed specifically for tracking the high-speed movements of a single person (e.g., fitness tracking, gesture control).

---

## 🖼️ 3. Image Classification Models
These models do not output coordinates. They analyze the *entire* image and output a single label describing the dominant object in the scene. *(Will crash if passed to SecurePi)*.

- **`imx500_network_mobilenet_v1_1.0_224_quant_pp.rpk`**
  - **Function:** MobileNet v1 classifier.
  - **Detects:** 1,000 specific ImageNet categories (e.g., dog breeds, car models, distinct plant species).

- **`imx500_network_efficientnet_lite0_int8_pp.rpk`**
  - **Function:** EfficientNet Lite0 classifier. Similar to MobileNet but often yields slightly higher accuracy.
  - **Detects:** 1,000 ImageNet categories.

---

## ✂️ 4. Semantic Segmentation Models
These models process the image pixel-by-pixel, coloring and grouping pixels into categories to create a mask. *(Will crash if passed to SecurePi)*.

- **`imx500_network_deeplabv3_mnv2_257x257_pp.rpk`**
  - **Function:** DeepLabV3. Used for foreground/background separation (like video call background blurring) or detailed scene analysis.
