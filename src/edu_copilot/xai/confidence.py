def calculate_prediction_confidence(prob: float) -> float:
    """
    Computes a normalized confidence score [0.0 to 1.0] indicating the model's 
    certainty, based on how far the probability is from the 0.5 decision boundary.
    
    Formula: Confidence = 2 * |prob - 0.5|
    
    Args:
        prob (float): Raw probability output from the ANN (sigmoid).
        
    Returns:
        float: Normalized confidence score.
    """
    return float(2.0 * abs(prob - 0.5))
