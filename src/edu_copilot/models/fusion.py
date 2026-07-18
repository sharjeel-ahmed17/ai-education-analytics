import numpy as np
from typing import Dict, Optional

def fuse_student_scores(
    tabular_prob: float,
    cnn_scores: Optional[Dict[str, float]] = None,
    timeseries_prob: Optional[float] = None,
    audio_score: Optional[float] = None,
    base_weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Fuses predictive risk probabilities and quality indicators from multiple modalities
    into a single composite recommendation score.
    
    Formula:
      Fused Score = w_tab * tabular_prob + w_cnn * cnn_risk + w_ts * timeseries_prob + w_aud * audio_risk
      where:
        cnn_risk = 1.0 - mean(legibility, correctness, completeness)
        audio_risk = 1.0 - audio_score
        
    Weights are dynamically normalized based on available modalities.
    
    Args:
        tabular_prob (float): Tabular ANN risk score [0.0, 1.0]. (Always required as core anchor).
        cnn_scores (dict): Dictionary with ['legibility', 'correctness', 'completeness'] keys.
        timeseries_prob (float): Forecasted failure risk from LSTM [0.0, 1.0].
        audio_score (float): Oral exam fluency score [0.0, 1.0].
        base_weights (dict): Prior weights for modalities. Default: Tabular=0.4, CNN=0.2, TS=0.2, Audio=0.2.
        
    Returns:
        float: Composite fused risk score in range [0.0, 1.0].
    """
    if base_weights is None:
        base_weights = {
            'tabular': 0.4,
            'cnn': 0.2,
            'timeseries': 0.2,
            'audio': 0.2
        }
        
    active_modalities = {'tabular': tabular_prob}
    active_weights = {'tabular': base_weights['tabular']}
    
    # Process CNN Image Modality
    if cnn_scores is not None and len(cnn_scores) > 0:
        legibility = cnn_scores.get('legibility', 1.0)
        correctness = cnn_scores.get('correctness', 1.0)
        completeness = cnn_scores.get('completeness', 1.0)
        
        # Calculate image-based quality
        cnn_quality = np.mean([legibility, correctness, completeness])
        # Convert positive quality to a risk metric
        active_modalities['cnn'] = 1.0 - float(cnn_quality)
        active_weights['cnn'] = base_weights['cnn']
        
    # Process Time Series Modality
    if timeseries_prob is not None:
        active_modalities['timeseries'] = timeseries_prob
        active_weights['timeseries'] = base_weights['timeseries']
        
    # Process Audio Modality
    if audio_score is not None:
        # Convert positive fluency score to a risk metric
        active_modalities['audio'] = 1.0 - audio_score
        active_weights['audio'] = base_weights['audio']
        
    # Re-normalize active weights so they sum to 1.0
    total_weight = sum(active_weights.values())
    normalized_weights = {k: v / total_weight for k, v in active_weights.items()}
    
    # Calculate fused composite risk score
    fused_score = sum(active_modalities[k] * normalized_weights[k] for k in active_modalities)
    
    return float(fused_score)
