"""Inference-only numeric exports; no cross-version sklearn unpickling."""
from pathlib import Path
import json
import numpy as np
from scipy.special import expit


class NumericModel:
    def __init__(self, path):
        self.kind = Path(path).stem
        self.params = json.loads(Path(path).read_text(encoding='utf-8'))

    def mahalanobis_distance_squared(self, X):
        delta = np.asarray(X, dtype=float) - np.asarray(self.params['location_'])
        return np.einsum('ij,jk,ik->i', delta, np.asarray(self.params['precision_']), delta)

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        p = self.params
        if self.kind == 'mahalanobis':
            values = -np.log1p(self.mahalanobis_distance_squared(X))
            score = values * p['coef'][0][0] + p['intercept'][0]
        elif self.kind == 'logistic':
            scaled = (X - np.asarray(p['mean'])) / np.asarray(p['scale'])
            score = scaled @ np.asarray(p['coef'][0]) + p['intercept'][0]
        elif self.kind == 'gradient_boosting':
            score = np.full(len(X), p['baseline'], dtype=float)
            for tree in p['trees']:
                node = np.zeros(len(X), dtype=int)
                leaves = np.asarray(tree['is_leaf'], dtype=bool)
                while True:
                    rows = np.flatnonzero(~leaves[node])
                    if len(rows) == 0:
                        break
                    current = node[rows]
                    value = X[rows, np.asarray(tree['feature_idx'])[current]]
                    go_left = value <= np.asarray(tree['num_threshold'])[current]
                    go_left = np.where(np.isnan(value), np.asarray(tree['missing_go_to_left'])[current], go_left)
                    node[rows] = np.where(go_left, np.asarray(tree['left'])[current], np.asarray(tree['right'])[current])
                score += np.asarray(tree['value'])[node]
        else:
            raise ValueError(f'Unknown numeric model: {self.kind}')
        probability = expit(score)
        return np.column_stack([1 - probability, probability])


class OnnxModel:
    def __init__(self, path):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])

    def predict_proba(self, X):
        return self.session.run(None, {self.session.get_inputs()[0].name: np.asarray(X, dtype=np.float32)})[1].astype(float)


def load_portable_models(directory):
    directory = Path(directory)
    return {
        'mahalanobis': NumericModel(directory / 'mahalanobis.json'),
        'logistic': NumericModel(directory / 'logistic.json'),
        'rbf_svm': OnnxModel(directory / 'rbf_svm.onnx'),
        'gradient_boosting': NumericModel(directory / 'gradient_boosting.json'),
    }
