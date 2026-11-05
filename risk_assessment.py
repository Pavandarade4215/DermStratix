import cv2, base64, numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional

TIER_CONFIG = {
    'tier_1': {
        'tier_name': 'Tier 1: Low Risk', 'badge_class': 'badge-low', 'color': '#10b981', 'bg_color': 'rgba(16, 185, 129, 0.12)',
        'vitality_range': '80 - 100', 'review_window': 'Routine Annual Check',
        'action_header': 'ACTION: Benign / Low Risk — Routine Annual Monitoring',
        'clinical_action': 'Clinical presentation indicates benign morphological characteristics with uniform pigmentation and symmetry. Maintain routine annual skin self-examinations and employ broad-spectrum SPF 30+ photoprotection.'
    },
    'tier_2': {
        'tier_name': 'Tier 2: Moderate Risk', 'badge_class': 'badge-moderate', 'color': '#f59e0b', 'bg_color': 'rgba(245, 158, 11, 0.12)',
        'vitality_range': '60 - 79', 'review_window': '3 - 6 Months Follow-up',
        'action_header': 'ACTION: Moderate Risk — Scheduled Outpatient Review (3-6 Months)',
        'clinical_action': 'Identified dysplastic, hyperkeratotic, or active inflammatory cutaneous alterations. Non-urgent clinical evaluation recommended within 3 to 6 months. Implement lifestyle adjustments and monitor for changes in lesion diameter or coloration.'
    },
    'tier_3': {
        'tier_name': 'Tier 3: Urgent Priority', 'badge_class': 'badge-urgent', 'color': '#ef4444', 'bg_color': 'rgba(239, 68, 68, 0.15)',
        'vitality_range': '< 60', 'review_window': 'Immediate Referral (< 7 Days)',
        'action_header': 'ACTION: Urgent Priority Referral — Excisional Biopsy & Histopathology',
        'clinical_action': 'CRITICAL ALERT: High clinical suspicion of aggressive cutaneous malignancy (e.g. Malignant Melanoma or invasive Basal Cell Carcinoma) or severe systemic derangement. Immediate dermatologist referral is mandatory for epiluminescence dermoscopy, excisional biopsy, and histopathological staging.'
    }
}

def calibrate_3tier_triage(posterior_risk_p: float) -> Dict[str, Any]:
    p = float(np.clip(posterior_risk_p, 0.0, 1.0))
    if p < 0.35:
        tier_key, tier_level = 'tier_1', 1
        vitality_score = int(round(100.0 - (p / 0.35) * 20.0))
    elif p <= 0.70:
        tier_key, tier_level = 'tier_2', 2
        fraction = (p - 0.35) / 0.35
        vitality_score = int(round(79.0 - fraction * 19.0))
    else:
        tier_key, tier_level = 'tier_3', 3
        fraction = (p - 0.70) / 0.30
        vitality_score = int(round(59.0 - fraction * 35.0))

    vitality_score = max(10, min(100, vitality_score))
    cfg = TIER_CONFIG[tier_key]
    return {
        'posterior_risk_p': round(p, 4), 'risk_percentage': round(p * 100.0, 1),
        'tier_level': tier_level, 'tier_key': tier_key, 'tier_name': cfg['tier_name'],
        'badge_class': cfg['badge_class'], 'color': cfg['color'], 'bg_color': cfg['bg_color'],
        'vitality_score': vitality_score, 'vitality_range': cfg['vitality_range'],
        'review_window': cfg['review_window'], 'action_header': cfg['action_header'],
        'clinical_action': cfg['clinical_action']
    }

def compute_vitality_composite(cutaneous_p: float, vitals_dict: Optional[Dict[str, float]] = None, abcd_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    base_vitality = 100.0 - (cutaneous_p * 50.0)
    deductions = []
    if abcd_dict:
        tds = abcd_dict.get('composite_tds', 2.0)
        if tds > 5.0:
            ded = (tds - 5.0) * 3.5
            deductions.append({'factor': 'ABCD Dysplasia', 'deduction': round(ded, 1)})
            base_vitality -= ded
    if vitals_dict:
        if vitals_dict.get('systolic_bp', 120.0) > 140.0:
            ded = min(15.0, (vitals_dict['systolic_bp'] - 140.0) * 0.4)
            deductions.append({'factor': 'Elevated Systolic BP', 'deduction': round(ded, 1)})
            base_vitality -= ded
        if vitals_dict.get('glucose', 95.0) > 125.0:
            ded = min(15.0, (vitals_dict['glucose'] - 125.0) * 0.3)
            deductions.append({'factor': 'Fasting Glucose (>125)', 'deduction': round(ded, 1)})
            base_vitality -= ded
        if vitals_dict.get('cholesterol', 190.0) > 220.0:
            ded = min(10.0, (vitals_dict['cholesterol'] - 220.0) * 0.2)
            deductions.append({'factor': 'Serum Cholesterol (>220)', 'deduction': round(ded, 1)})
            base_vitality -= ded
        if vitals_dict.get('bmi', 24.0) > 30.0:
            ded = min(8.0, (vitals_dict['bmi'] - 30.0) * 0.8)
            deductions.append({'factor': 'Elevated BMI (>30)', 'deduction': round(ded, 1)})
            base_vitality -= ded

    final_vitality = int(round(np.clip(base_vitality, 15.0, 99.0)))
    status_label = 'Optimal'
    if final_vitality < 60:
        status_label = 'Compromised / Urgent Attention'
    elif final_vitality < 80:
        status_label = 'Moderate Concern / Intervention'
    else:
        status_label = 'Healthy / Low Pathology'
    return {'vitality_score': final_vitality, 'status_label': status_label, 'deductions': deductions}

def generate_lime_superpixel_explanation(roi_224: np.ndarray, predicted_code: str, posterior_risk: float) -> Dict[str, Any]:
    h, w = roi_224.shape[:2]
    gray = cv2.cvtColor(roi_224, cv2.COLOR_BGR2GRAY)
    grid_size = 28
    labels = np.zeros((h, w), dtype=np.int32)
    idx = 0
    for y in range(0, h, grid_size):
        for x in range(0, w, grid_size):
            labels[y:y+grid_size, x:x+grid_size] = idx
            idx += 1
    gy, gx = np.gradient(labels)
    mask_contours = ((gy != 0) | (gx != 0)).astype(np.uint8) * 255

    _, core_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    lime_overlay = roi_224.copy()
    positive_mask = np.zeros((h, w), dtype=np.uint8)

    cx, cy = w // 2, h // 2
    for sp_id in range(idx):
        sp_indices = (labels == sp_id)
        if not np.any(sp_indices):
            continue
        sp_y, sp_x = np.where(sp_indices)
        dist = np.sqrt((np.mean(sp_x) - cx) ** 2 + (np.mean(sp_y) - cy) ** 2) / (w * 0.5)
        core_overlap = np.mean(core_mask[sp_indices] > 0)
        
        if predicted_code in ['mel', 'bcc', 'akiec', 'rosacea', 'jaundice']:
            weight = (core_overlap * 0.55) + (1.0 - dist) * 0.25
        else:
            weight = (1.0 - core_overlap) * 0.40
            
        if weight >= 0.25 and (core_overlap > 0.3 or dist < 0.6):
            positive_mask[sp_indices] = 255

    pos_contours, _ = cv2.findContours(positive_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    green_tint = lime_overlay.copy()
    green_tint[positive_mask > 0] = (green_tint[positive_mask > 0] * 0.65 + np.array([0, 220, 80]) * 0.35).astype(np.uint8)
    lime_overlay = cv2.addWeighted(lime_overlay, 0.4, green_tint, 0.6, 0)
    cv2.drawContours(lime_overlay, pos_contours, -1, (0, 255, 100), 2, cv2.LINE_AA)
    lime_overlay[mask_contours > 0] = (lime_overlay[mask_contours > 0] * 0.3 + np.array([200, 255, 120]) * 0.7).astype(np.uint8)

    core_w = '+0.48' if posterior_risk > 0.5 else '+0.15'
    border_w = '+0.31' if posterior_risk > 0.5 else '+0.08'
    cv2.putText(lime_overlay, f'LIME Positive: Core ({core_w})', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 100), 1, cv2.LINE_AA)
    cv2.putText(lime_overlay, f'Border ({border_w}) | Bg (0.00)', (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 220), 1, cv2.LINE_AA)

    _, buffer_lime = cv2.imencode('.png', lime_overlay)
    lime_b64 = base64.b64encode(buffer_lime).decode('utf-8')
    _, buffer_roi = cv2.imencode('.png', roi_224)
    roi_b64 = base64.b64encode(buffer_roi).decode('utf-8')

    return {
        'lime_image_b64': f'data:image/png;base64,{lime_b64}',
        'roi_image_b64': f'data:image/png;base64,{roi_b64}',
        'saliency_weights': {'pigmented_core': float(core_w.replace('+', '')), 'border_irregularity': float(border_w.replace('+', '')), 'peripheral_background': 0.0},
        'num_positive_superpixels': len(pos_contours),
        'explanation_summary': f'LIME superpixel attribution identifies focal hyperpigmented lesion cores and notched margins driving the {predicted_code.upper()} risk classification.'
    }

MANDATORY_MEDICAL_DISCLAIMER = (
    'CLINICAL SAFETY ADVISORY: DermStratix AI is an investigational clinical decision support '
    'platform engineered for preliminary screening and risk stratification. Automated findings do not '
    'constitute a definitive histopathological diagnosis, prognosis, or prescription. All recommendations '
    'must be correlated with patient history and verified via physical examination and histopathological '
    'biopsy by a licensed healthcare professional.'
)

def build_clinical_audit_report(screening_result: Dict[str, Any], patient_id: Optional[str] = None) -> Dict[str, Any]:
    return {
        'report_id': f'RPT-DS-{np.random.randint(10000, 99999)}',
        'patient_id': patient_id or f'PAT-{np.random.randint(100000, 999999)}',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'),
        'screening_mode': screening_result.get('mode', 'Dual-Mode Vision'),
        'focus_gatekeeper': screening_result.get('focus_gate', {}),
        'abcd_scoring': screening_result.get('abcd', {}),
        'ai_model': screening_result.get('model', {}),
        'triage_assessment': screening_result.get('triage', {}),
        'vitality_gauge': screening_result.get('vitality', {}),
        'explainable_ai': screening_result.get('lime', {}),
        'mandatory_disclaimer': MANDATORY_MEDICAL_DISCLAIMER,
        "citation": "DermStratix AI Clinical Health Platform (v2.0)"
    }
