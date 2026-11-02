"""Read-only dependency, calibration and pretrained-model checks."""
import importlib
import importlib.metadata
from pathlib import Path
import sys

PACKAGES = {'numpy': 'numpy', 'pandas': 'pandas', 'scipy': 'scipy', 'scikit-learn': 'sklearn', 'matplotlib': 'matplotlib', 'joblib': 'joblib', 'PyYAML': 'yaml', 'Pillow': 'PIL', 'tifffile': 'tifffile', 'tabulate': 'tabulate', 'onnxruntime': 'onnxruntime'}

def main():
    print('Python:', sys.executable)
    print('Version:', sys.version.split()[0])
    missing = []
    for package, module in PACKAGES.items():
        try:
            importlib.import_module(module)
            print(f'OK {package} {importlib.metadata.version(package)}')
        except Exception as exc:
            print(f'FAIL {package}: {exc}')
            missing.append(package)
    if missing:
        print('Install missing packages in the active smol_env:')
        print('python -m pip install ' + ' '.join(missing))
        return 1
    from rrl_track_behaviour_ml.config import load_config
    from rrl_track_behaviour_ml.replacement_reference import validate_replacement_reference
    root = Path(__file__).resolve().parent
    try:
        config = load_config(root / 'config_H2B_reference.yaml', root / 'RRL_user_settings.yaml')
        result = validate_replacement_reference(config['paths']['output_dir'], config)
        print('PASS:', result['release_name'])
        print('All four portable models loaded. No packages or settings were changed.')
    except Exception as exc:
        print(f'FAIL reference: {exc}')
        return 1
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
