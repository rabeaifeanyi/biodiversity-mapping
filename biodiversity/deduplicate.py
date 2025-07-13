import numpy as np

def deduplicate_points(points, distance_threshold=0.5):
    """
    Removes duplicate detections that are closer than distance_threshold.
    
    Args:
        points (list of tuples): Each tuple is (x, y, class, confidence).
        distance_threshold (float): Minimum allowed distance in meters.
    
    Returns:
        list of tuples: Deduplicated points.
    """
    if not points:
        return []

    points_array = np.array([[p[0], p[1]] for p in points])
    keep_indices = set(range(len(points)))

    dists = np.linalg.norm(points_array[:, None, :] - points_array[None, :, :], axis=-1)

    for i in range(len(points)):
        if i not in keep_indices:
            continue
        for j in range(i + 1, len(points)):
            if j not in keep_indices:
                continue
            if dists[i, j] < distance_threshold:

                if points[i][3] >= points[j][3]:
                    keep_indices.discard(j)
                else:
                    keep_indices.discard(i)
                    break

    deduped = [points[i] for i in sorted(keep_indices)]
    return deduped

