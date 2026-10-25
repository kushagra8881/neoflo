import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

class ClusterEngine:
    """
    Unsupervised document clustering engine using K-Means, automatic K-estimation,
    silhouette quality diagnostics, ambiguity detection, and 2D PCA projection.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def estimate_best_k(
        self, 
        embeddings: np.ndarray, 
        min_k: int = 2, 
        max_k: int = 8
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Evaluates candidate values of K using Silhouette Score and Davies-Bouldin Index.
        Returns the optimal K and an evaluation history table.
        """
        n_samples = len(embeddings)
        if n_samples <= 1:
            return 1, {"scores": {1: {"silhouette": 1.0, "davies_bouldin": 0.0}}}

        if n_samples == 2:
            # For 2 samples, test if they are similar or distinct
            sim = float(np.dot(embeddings[0], embeddings[1]))
            chosen_k = 1 if sim >= 0.70 else 2
            return chosen_k, {"scores": {chosen_k: {"similarity": round(sim, 3)}}}

        upper_bound = min(max_k, n_samples - 1)
        if upper_bound < min_k:
            return min_k, {"scores": {}}

        scores = {}
        best_k = min_k
        best_score = float("-inf")

        for k in range(min_k, upper_bound + 1):
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(embeddings)

            # Ensure we have at least 2 distinct clusters formed
            if len(set(labels)) < 2:
                continue

            try:
                sil = float(silhouette_score(embeddings, labels))
                db = float(davies_bouldin_score(embeddings, labels))
                ch = float(calinski_harabasz_score(embeddings, labels))
            except Exception:
                sil, db, ch = 0.0, 99.0, 0.0

            scores[k] = {
                "silhouette": round(sil, 4),
                "davies_bouldin": round(db, 4),
                "calinski_harabasz": round(ch, 2)
            }

            if sil > best_score:
                best_score = sil
                best_k = k

        return best_k, {"scores": scores, "best_silhouette": round(best_score, 4)}

    def cluster(
        self, 
        embeddings: np.ndarray, 
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Runs K-Means clustering, computes quality metrics, ambiguity flags, and 2D PCA.
        """
        n_samples = len(embeddings)
        if n_samples == 0:
            return {"k": 0, "labels": [], "clusters": {}}

        if n_samples == 1:
            return {
                "k": 1,
                "labels": [0],
                "quality": {"silhouette": 1.0, "status": "Single document"},
                "ambiguity": [False],
                "pca_coords": [[0.0, 0.0]],
                "clusters": {0: {"indices": [0], "size": 1}}
            }

        # 1. Determine optimal K if not explicitly provided
        best_k_eval = {}
        if k is None or k <= 0:
            k, best_k_eval = self.estimate_best_k(embeddings)
            k = max(1, min(k, n_samples))

        if k == 1:
            pca_coords = self._compute_pca(embeddings)
            return {
                "k": 1,
                "labels": [0] * n_samples,
                "quality": {"silhouette": 1.0, "status": "Single unified cluster"},
                "ambiguity": [False] * n_samples,
                "pca_coords": pca_coords,
                "clusters": {0: {"indices": list(range(n_samples)), "size": n_samples}}
            }

        # 2. Fit K-Means
        kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=15)
        labels = kmeans.fit_predict(embeddings)
        centroids = kmeans.cluster_centers_

        # Normalize centroids
        centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        centroid_norms[centroid_norms == 0] = 1.0
        normalized_centroids = centroids / centroid_norms

        # 3. Calculate distance to centroids & ambiguity flags
        # Euclidean distance matrix from each sample to each centroid
        dist_matrix = np.linalg.norm(embeddings[:, np.newaxis, :] - normalized_centroids[np.newaxis, :, :], axis=2)

        distances_to_assigned = []
        is_ambiguous = []
        for i in range(n_samples):
            assigned_c = labels[i]
            d_assigned = float(dist_matrix[i, assigned_c])
            distances_to_assigned.append(round(d_assigned, 4))

            # Sort distances to all centroids
            sorted_dists = np.sort(dist_matrix[i])
            # If 2nd closest centroid is very close in distance (ratio < 1.18), flag as ambiguous
            if len(sorted_dists) > 1:
                ratio = sorted_dists[1] / max(0.001, sorted_dists[0])
                ambiguous = bool(ratio < 1.18 or d_assigned > 0.85)
            else:
                ambiguous = False
            is_ambiguous.append(ambiguous)

        # 4. Cluster quality metrics
        try:
            sil_overall = float(silhouette_score(embeddings, labels))
            sample_silhouettes = silhouette_samples(embeddings, labels)
            db_score = float(davies_bouldin_score(embeddings, labels))
        except Exception:
            sil_overall = 0.0
            sample_silhouettes = np.zeros(n_samples)
            db_score = 0.0

        # Quality badge rating
        if sil_overall >= 0.45:
            quality_status = "Good"
        elif sil_overall >= 0.20:
            quality_status = "Moderate"
        else:
            quality_status = "Review Recommended"

        # 5. 2D PCA Projection for scatter plot visualization
        pca_coords = self._compute_pca(embeddings)

        # 6. Group document indices per cluster
        clusters = {}
        for c_id in range(k):
            indices = np.where(labels == c_id)[0].tolist()
            clusters[c_id] = {
                "cluster_id": c_id,
                "size": len(indices),
                "indices": indices,
                "avg_distance": round(float(np.mean([distances_to_assigned[idx] for idx in indices])), 4) if indices else 0.0
            }

        return {
            "k": k,
            "labels": labels.tolist(),
            "quality": {
                "silhouette_score": round(sil_overall, 4),
                "davies_bouldin_score": round(db_score, 4),
                "status": quality_status,
                "k_evaluation": best_k_eval
            },
            "distances": distances_to_assigned,
            "sample_silhouettes": [round(float(s), 4) for s in sample_silhouettes],
            "is_ambiguous": is_ambiguous,
            "pca_coords": pca_coords,
            "clusters": clusters
        }

    def _compute_pca(self, embeddings: np.ndarray) -> List[List[float]]:
        """Projects high-dimensional embeddings to 2D coordinates for visual plotting."""
        n_samples = len(embeddings)
        if n_samples == 0:
            return []
        if n_samples == 1:
            return [[0.0, 0.0]]
        if n_samples == 2:
            return [[-1.0, 0.0], [1.0, 0.0]]

        pca = PCA(n_components=2, random_state=self.random_state)
        coords = pca.fit_transform(embeddings)
        return [[round(float(pt[0]), 4), round(float(pt[1]), 4)] for pt in coords]
