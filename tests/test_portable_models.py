import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from rrl_track_behaviour_ml.models import load_models, consensus_labels
from rrl_track_behaviour_ml.config import load_config
from rrl_track_behaviour_ml.replacement_reference import validate_replacement_reference

ROOT = Path(__file__).resolve().parents[1]

class PortableModelsTests(unittest.TestCase):
    def test_export_matches_original_fitted_models_on_synthetic_features(self):
        config = load_config(ROOT / 'config_H2B_reference.yaml', ROOT / 'RRL_user_settings.yaml')
        models = load_models(Path(config['paths']['output_dir']) / 'models')
        fixture = np.load(Path(__file__).parent / 'fixtures' / 'portable_model_regression.npz')
        actual = {}
        for name, model in models.items():
            actual[name] = model.predict_proba(fixture['X'])[:,1]
            np.testing.assert_allclose(actual[name], fixture[name], rtol=0, atol=5e-5)
        settings = config['model']
        before = consensus_labels(pd.DataFrame({f'probability_{k}': fixture[k] for k in models}), settings)
        after = consensus_labels(pd.DataFrame({f'probability_{k}': actual[k] for k in models}), settings)
        self.assertEqual(before.consensus_prediction.tolist(), after.consensus_prediction.tolist())

    def test_mismatched_acquisition_is_rejected(self):
        config = load_config(ROOT / 'config_H2B_reference.yaml', ROOT / 'RRL_user_settings.yaml')
        config['feature_calculation']['pixel_size_um'] = .1
        with self.assertRaisesRegex(ValueError, 'pixel size'):
            validate_replacement_reference(config['paths']['output_dir'], config)

if __name__ == '__main__':
    unittest.main()
