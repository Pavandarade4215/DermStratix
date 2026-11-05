# DermStratix AI™ — Healthcare Decision Support Platform
### AI-Assisted Facial & Skin Health Screening

DermStratix AI is a clinical-grade web application providing automated computer vision decision support for dermatological skin lesions and facial cutaneous conditions with calibrated multi-tier risk stratification and explainability.

---

## 🏥 Platform Capabilities

1. **Dual-Mode Clinical Acquisition**:
   - **Dermoscopic Skin Lesion Screening**: Close-up epiluminescence analysis of pigmented skin lesions, melanocytic nevi, melanoma, and carcinomas.
   - **Facial Skin Screening**: Frontal portrait screening for vascular erythema (rosacea), inflammatory acne vulgaris, melasma hyperpigmentation, and systemic bilirubin jaundice markers.
2. **Laplacian Optical Focus Gatekeeper**:
   - Verifies optical sharpness before inference: Var(∇²I) >= 30.0.
   - Automatically applies CLAHE (Contrast Limited Adaptive Histogram Equalization) contrast normalization on soft-focus images.
3. **Automated ABCD Dermatological Rubric**:
   - **A (Asymmetry Index)**: Dual-axis contour eccentricity and symmetry divergence (0–10).
   - **B (Border Irregularity)**: Compactness ratio P² / (4πA) identifying notched margins (0–10).
   - **C (Color Variegation)**: Multi-spectral variance across RGB, HSV, and CIELAB color spaces (0–10).
   - **D (Diameter)**: Calibrated millimeter caliber dimension (> 6.0 mm warning threshold).
4. **Calibrated 3-Tier Clinical Risk Triage**:
   - **Tier 1: Low Risk** (P < 0.35): Routine annual monitoring recommended.
   - **Tier 2: Moderate Risk** (0.35 <= P <= 0.70): Scheduled outpatient clinical follow-up within 3–6 months.
   - **Tier 3: Urgent Priority** (P > 0.70): Specialist dermatologist referral and excisional biopsy recommended.
5. **Explainable AI (LIME)**:
   - Superpixel visual attribution highlighting localized lesion regions and notched margins driving classification.
6. **Patient Screening History & Export**:
   - Persistent screening logs with date, findings, triage status, and PDF-ready summary reports.

---

## 🚀 Quickstart & Execution

```bash
# 1. Open project folder
cd DermStratix_AI_Project

# 2. Run the platform server
python run.py
# or
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000/
```

---

## ⚖️ Clinical Safety Advisory
DermStratix AI is an investigational clinical decision support platform engineered for preliminary risk stratification and triaging. Automated findings do not constitute a definitive histopathological diagnosis, medical prognosis, or prescription. All screening assessments must be correlated with patient clinical history and verified via physical examination and histopathological biopsy by a licensed healthcare professional.
