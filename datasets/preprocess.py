# preprocess.py (Conceptual logic for OralCavity data processing)
import numpy as np

def hybrid_normalization(partial, gt):
    # 70% based on GT, 30% based on partial input
    if np.random.rand() < 0.7:
        center = np.mean(gt, axis=0)
        scale = np.max(np.linalg.norm(gt - center, axis=1))
    else:
        center = np.mean(partial, axis=0)
        scale = np.max(np.linalg.norm(partial - center, axis=1))
    return (partial - center) / scale, (gt - center) / scale

def symmetry_enhancement(points, axis=0):
    # Left-right symmetry enhancement
    flipped = points.copy()
    flipped[:, axis] = -flipped[:, axis]
    return np.concatenate([points, flipped], axis=0)

def noise_injection(partial, tooth_mask=None):
    # Noise injection avoiding tooth regions
    if tooth_mask is None:
        tooth_mask = np.zeros(partial.shape[0], dtype=bool)
    noise = np.random.normal(0, 0.01, partial.shape)
    partial[~tooth_mask] += noise[~tooth_mask]
    return partial
