"""Maintainer-only export from trusted original joblib files, without refitting.

Requires the historical sklearn 1.4.2 environment plus skl2onnx and onnx.
Do not install that training stack into the user's smol_env. Normal users
consume the already exported models and never need this script.
"""
from pathlib import Path
import hashlib
import json
import sys
import joblib
import numpy as np
import sklearn


def main():
    if sklearn.__version__ != '1.4.2':
        raise RuntimeError('Export requires the historical scikit-learn 1.4.2 environment')
    from skl2onnx import to_onnx
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    directory = next((root / 'outputs').glob('Trainingset-*')) / 'models'
    for name in ['mahalanobis', 'logistic', 'rbf_svm', 'gradient_boosting']:
        model = joblib.load(directory / f'RRL_{name}_model.joblib')
        if name == 'rbf_svm':
            graph = to_onnx(model, np.zeros((1,4), np.float32), target_opset=17, options={id(model): {'zipmap': False}})
            (directory / 'rbf_svm.onnx').write_bytes(graph.SerializeToString())
            continue
        if name == 'mahalanobis':
            params = {key: getattr(model, key).tolist() for key in ['location_', 'precision_']}
            params.update(coef=model.calibrator_.coef_.tolist(), intercept=model.calibrator_.intercept_.tolist())
        elif name == 'logistic':
            scaler, estimator = model.steps[0][1], model.steps[1][1]
            params = dict(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(), coef=estimator.coef_.tolist(), intercept=estimator.intercept_.tolist())
        else:
            trees = []
            for stage in model._predictors:
                if len(stage) != 1 or stage[0].nodes['is_categorical'].any():
                    raise ValueError('Export supports binary models with numeric splits only')
                trees.append({key: stage[0].nodes[key].tolist() for key in ['value', 'feature_idx', 'num_threshold', 'missing_go_to_left', 'left', 'right', 'is_leaf']})
            params = dict(baseline=float(model._baseline_prediction[0,0]), trees=trees)
        (directory / f'{name}.json').write_text(json.dumps(params), encoding='utf-8')
    manifest_path = directory / 'PORTABLE_MODELS.json'
    metadata = json.loads(manifest_path.read_text(encoding='utf-8'))
    for name in metadata['sha256']:
        metadata['sha256'][name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Exports written. Run parity tests and regenerate distribution checksums before publishing.')


if __name__ == '__main__':
    main()
