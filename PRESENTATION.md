# 🎤 GreenLens — Official Project Presentation Guide (Slide-by-Slide)

> **Audience:** Project Managers, Agronomists, and Technical Reviewers.  
> **Speaker Role:** Person 4 (Integration & Presentation Lead) with contributions from Persons 1, 2, and 3.  
> **Estimated Duration:** 10–12 minutes + Live Demo.

---

##  Slide 1: Title & Executive Summary
- **Title:** GreenLens — Real-Time Plant Leaf Disease Diagnosis Using Explainable AI
- **Subtitle:** An End-to-End Deep Learning System for Precision Agriculture
- **Core Hook:** *"Plant diseases reduce global crop yields by 20–40% annually. GreenLens brings expert-level plant pathology into the hands of farmers in less than 2 milliseconds."*
- **Key Talking Points:**
  - 15 classes covering Tomato, Potato, and Pepper Bell.
  - **99.66% Accuracy** on unseen held-out test data.
  - Lightweight (42 MB) and edge-deployable.
  - Bilingual agronomic guidance (Arabic & English).

---

##  Slide 2: The Problem Statement & Agricultural Reality
- **Field Challenges:**
  - Expert plant pathologists cannot visit every smallholder farm.
  - Visual symptom confusion: Diseases like *Early Blight*, *Late Blight*, and *Target Spot* look nearly identical in early stages.
  - Delayed treatment leads to total crop loss.
- **Why Standard AI Fails in Agriculture:**
  - Most models output only a technical label (e.g. `Tomato___Early_blight`) without telling the farmer what to do next.
  - Lack of trust: Deep learning models are treated as black boxes.
  - Class imbalance in real field datasets leads to high false-negative rates.

---

##  Slide 3: Team Division & Modular Architecture (4 Person Roles)
- **Person 1 (AI / Modeling):** Dataset curation, Preprocessing, ResNet-18 architecture, and advanced training pipeline.
- **Person 2 (Web / UX):** Responsive Streamlit app, bilingual RTL Arabic UI, camera/upload inputs.
- **Person 3 (Agronomic Knowledge):** Comprehensive bilingual database (`disease_info.json`) linking symptoms, pathogens, treatments, and preventions.
- **Person 4 (Integration & Quality Assurance):** Unified inference engine (`predict.py`), Grad-CAM explainability, latency benchmarking, and automated test suite.

---

##  Slide 4: Dataset Deep Dive & The Imbalance Challenge
- **Dataset:** PlantVillage (20,624 total images).
- **Stratified Split:** 16,499 Train (80%) | 2,062 Val (10%) | 2,063 Test (10%).
- **The Imbalance Challenge:**
  - Majority class: *Tomato Yellow Leaf Curl* (2,566 images).
  - Minority class: *Potato Healthy* (122 images) and *Tomato Mosaic Virus* (298 images).
  - In our baseline, **Tomato Early Blight had a 20% miss rate (Recall = 80%)** due to class imbalance and visual overlap with Septoria.

---

##  Slide 5: The AI Breakthrough (Baseline vs. Improved Model)
- **Why not ResNet-50 or ViT?** ResNet-18 offered the best Pareto frontier of parameter count (11.18M), latency (1.5ms), and accuracy.
- **The 4 Key Engineering Interventions:**
  1. **Domain-Specific Augmentations:** `ColorJitter` (lighting variations) + `RandomVerticalFlip` + 15° Rotation.
  2. **Differential Learning Rates:** Backbone at `1e-4` (preserves ImageNet features), Head at `1e-3` (learns 15 target classes fast).
  3. **Smoothed Inverse-Frequency Loss:** Mathematically penalizes errors on rare and hard classes without gradient instability.
  4. **CosineAnnealingLR Scheduler:** Smooth learning rate decay that eradicated the late-stage overfitting seen in the baseline.
- **The Result:**
  - Validation Accuracy: **96.85% ➔ 99.56%** (+2.71%)
  - Early Blight Recall: **80.00% ➔ 98.00%** (+18.00% jump!)
  - Macro F1: **0.9680 ➔ 0.9958**

---

##  Slide 6: Final Held-Out Test Set Verification
- **Unbiased Real-World Test (2,063 completely unseen images):**
  - **Test Overall Accuracy:** **99.66%**
  - **Test Macro F1-Score:** **0.9943**
  - **Test Balanced Accuracy:** **99.20%**
  - **Total Errors:** Only **7 images misclassified out of 2,063**!
  - **Zero Catastrophic False Negatives:** All healthy samples and critical blights identified with >99% precision.

---

##  Slide 7: Hardware & Latency Benchmarks (Ready for the Edge)
- **Benchmarking Rigor:** Excluded 20 warmup cycles; measured 100 runs using `torch.cuda.synchronize()`.
- **Latency Numbers:**
  - **GPU (RTX 4050, Batch 1):** **1.53 ms** (653.5 FPS)
  - **GPU (RTX 4050, Batch 32):** **0.67 ms / image** (1,488 FPS)
  - **CPU (Single-Core, Batch 1):** **16.24 ms** (61.6 FPS)
- **Deployment Footprint:**
  - Model weights: **42.74 MB**
  - FLOPs: **1.82 GFLOPs**
  - Training cost: **~2.2 minutes total** on a single laptop GPU (virtually free cloud compute).

---

##  Slide 8: Trust & Explainable AI (Grad-CAM)
- **Explainability in Action:**
  - Uses Gradient-weighted Class Activation Mapping on `layer4` of ResNet-18.
  - Highlights specific leaf lesions, necrotic spots, and viral chlorosis.
  - Proves the model is learning biological pathology rather than background soil, watermarks, or leaf edges.

---

##  Slide 9: Agronomic Impact & The Knowledge Base
- The model prediction is only the first step.
- `DiseaseDB` immediately translates the diagnosis into:
  - **Pathogen Cause** (fungal, bacterial, viral).
  - **Severity Rating** (Moderate, High, Very High).
  - **Field Symptoms Checklist**.
  - **Actionable Treatment Protocol** (immediate cultural/chemical recommendations).
  - **Long-Term Prevention** (crop rotation, irrigation management).
- Full bilingual support in **Arabic** and **English** ensures accessibility for regional extension officers.

---

##  Slide 10: Live Demonstration (Interactive Streamlit Demo)
- **Walkthrough Steps:**
  1. Show bilingual toggle with Arabic Right-to-Left (RTL) formatting.
  2. Select a preloaded sample (e.g., *Tomato Bacterial Spot*).
  3. Click **"Analyze & Diagnose"**.
  4. Observe the sub-second prediction and confidence metric (100%).
  5. Toggle the **Grad-CAM Attention Map** to show the glowing heatmap over leaf lesions.
  6. Expand the **Treatment Plan** and **Prevention** sections.

---

##  Slide 11: Future Roadmap & Production Scaling
1. **Model Optimization:** Quantization to INT8 (reducing model size from 42 MB to ~11 MB) for Flutter mobile deployment.
2. **Object Detection (YOLOv8/v11):** Multi-leaf detection directly from canopy drone photos.
3. **Sensor Fusion:** Integrating local weather data (humidity, temperature) to predict fungal outbreak risk before symptoms appear.

---

##  Slide 12: Q&A Anticipated Questions
- **Q1: What if a farmer uploads a photo of an apple leaf or a non-plant object?**  
  *Answer:* GreenLens includes an **Uncertainty Detector** (`confidence < 60%`), which flags low-confidence or out-of-domain images and advises the user to retake the photo.
- **Q2: Why not train on more epochs?**  
  *Answer:* With differential learning rates and CosineAnnealingLR, the validation loss dropped to 0.0113 by epoch 6, achieving 99.66% test accuracy. Training further risked overfitting.
- **Q3: Can this run in fields with no cellular internet?**  
  *Answer:* Yes. With 16 ms CPU latency and a 42 MB footprint, the model runs entirely on-device without cloud connectivity.
