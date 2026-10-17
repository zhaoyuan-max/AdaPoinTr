# preprocess.py (Conceptual logic for OralCavity data processing)
import numpy as np

def hybrid_normalization(partial, gt, is_training=True):
    """
    Hybrid normalization strategy. 
    Training: 70% based on GT, 30% based on partial.
    Validation/Test: 100% based on partial (defective).
    """
    if is_training and np.random.rand() < 0.7:
        # Training: 70% based on GT
        center = np.mean(gt, axis=0)
        scale = np.max(np.linalg.norm(gt - center, axis=1))
    else:
        # Training: 30%; Validation/Test: 100% based on partial
        center = np.mean(partial, axis=0)
        scale = np.max(np.linalg.norm(partial - center, axis=1))
        
    scale = max(scale, 1e-8)
    return (partial - center) / scale, (gt - center) / scale

def symmetry_enhancement(partial, gt, axis=0):
    """
    Left-right symmetry augmentation along a specific axis.
    """
    partial_flip = partial.copy()
    partial_flip[:, axis] = -partial_flip[:, axis]
    
    gt_flip = gt.copy()
    gt_flip[:, axis] = -gt_flip[:, axis]
    
    return np.concatenate([partial, partial_flip], axis=0), np.concatenate([gt, gt_flip], axis=0)

def noise_injection(partial, defect_mask, is_training=True):
    """
    Zero-mean Gaussian noise (sigma=0.01 per axis) added to defect mask only 
    with a probability of 0.5.
    """
    if not is_training or np.random.rand() > 0.5:
        return partial
        
    noise = np.random.normal(0, 0.01, partial.shape)
    partial_noisy = partial.copy()
    partial_noisy[defect_mask] += noise[defect_mask]
    return partial_noisy
