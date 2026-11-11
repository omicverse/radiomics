"""Optional SimpleITK / PyRadiomics adapters; no image dependencies at import time."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from ._validation import unique_ids


def _sitk():
    try:
        import SimpleITK
        return SimpleITK
    except ImportError as exc:
        raise ImportError('Install ov-radiomics[image] to read medical images.') from exc


def match_images(root, image_dir="images", mask_dir="masks"):
    """Pair .nii/.nii.gz files by exact filename. IDs retain the filename."""
    root = Path(root)
    def files(directory):
        folder = root / directory
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        return {p.name: p.resolve() for p in folder.iterdir()
                if p.is_file() and (p.name.endswith('.nii') or p.name.endswith('.nii.gz'))}
    images, masks = files(image_dir), files(mask_dir)
    if not images or images.keys() != masks.keys():
        raise ValueError(f"Image/mask names differ or are empty. Missing masks: {sorted(images.keys()-masks.keys())}; missing images: {sorted(masks.keys()-images.keys())}")
    return pd.DataFrame([{'ID': key, 'image': str(images[key]), 'mask': str(masks[key])}
                         for key in sorted(images)])


def check_images(samples, roi_label=1, geometry_tolerance=1e-6):
    """Read each pair and report geometry mismatch and absent ROI without modifying files."""
    unique_ids(samples)
    sitk = _sitk()
    rows = []
    for row in samples.to_dict('records'):
        result = {'ID': row['ID'], 'ok': False, 'error': ''}
        try:
            image, mask = sitk.ReadImage(str(row['image'])), sitk.ReadImage(str(row['mask']))
            if image.GetDimension() != mask.GetDimension():
                raise ValueError('Image/mask dimensions differ.')
            mismatch = []
            for name in ('Size', 'Spacing', 'Origin', 'Direction'):
                a, b = getattr(image, 'Get'+name)(), getattr(mask, 'Get'+name)()
                equal = a == b if name == 'Size' else np.allclose(a, b, atol=geometry_tolerance, rtol=0)
                if not equal:
                    mismatch.append(name)
            count = int(np.count_nonzero(sitk.GetArrayViewFromImage(mask) == row.get('roi_label', roi_label)))
            result.update(roi_voxels=count, geometry_mismatch=','.join(mismatch))
            if count == 0:
                raise ValueError('Requested ROI is empty or absent.')
            if mismatch:
                raise ValueError('Geometry mismatch: ' + ', '.join(mismatch))
            result['ok'] = True
        except Exception as exc:
            result['error'] = str(exc)
        rows.append(result)
    return pd.DataFrame(rows)


@dataclass
class ExtractionResult:
    features: pd.DataFrame
    diagnostics: pd.DataFrame
    config: dict

    def save(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self.features.to_csv(path / 'features.csv', index=False)
        self.diagnostics.to_csv(path / 'diagnostics.csv', index=False)
        (path / 'config.json').write_text(json.dumps(self.config, indent=2, default=str), encoding='utf-8')


def extract(samples, params=None, roi_label=1, correct_mask=False, n_jobs=1, on_error="raise"):
    """Extract ROI features. params is a PyRadiomics YAML path or parameter dictionary.

    Explicit correct_mask overrides the backend setting. No implicit disk cache.
    on_error='record' keeps failed IDs in diagnostics and features (NaN values).
    """
    unique_ids(samples)
    if on_error not in ('raise', 'record') or samples.empty:
        raise ValueError("Use nonempty samples and on_error='raise' or 'record'.")
    try:
        import radiomics
        from radiomics.featureextractor import RadiomicsFeatureExtractor
    except ImportError as exc:
        raise ImportError('Install ov-radiomics[image] for extraction.') from exc
    import copy
    if isinstance(params, (str, Path)):
        import yaml
        config = yaml.safe_load(Path(params).read_text(encoding='utf-8'))
    else:
        config = copy.deepcopy(params)
    template = RadiomicsFeatureExtractor(config, correctMask=correct_mask) if config is not None else RadiomicsFeatureExtractor(correctMask=correct_mask)
    effective = {'setting': template.settings, 'imageType': template.enabledImagetypes,
                 'featureClass': template.enabledFeatures}

    def one(row):
        features, diagnostic = {'ID': row['ID']}, {'ID': row['ID'], 'ok': False}
        try:
            qc = check_images(pd.DataFrame([row]), roi_label=roi_label).iloc[0]
            if not qc['ok'] and not (correct_mask and str(qc['error']).startswith('Geometry mismatch:')):
                raise ValueError(qc['error'])
            extractor = RadiomicsFeatureExtractor(copy.deepcopy(effective))
            output = extractor.execute(str(row['image']), str(row['mask']), label=int(row.get('roi_label', roi_label)))
            for key, value in output.items():
                if key.startswith('diagnostics_'):
                    diagnostic[key] = str(value)
                else:
                    features[key] = float(value)
            diagnostic.update(ok=True, error='')
        except Exception as exc:
            if on_error == 'raise':
                raise RuntimeError(f"Extraction failed for {row['ID']}: {exc}") from exc
            diagnostic['error'] = str(exc)
        return features, diagnostic
    results = Parallel(n_jobs=n_jobs)(delayed(one)(row) for row in samples.to_dict('records'))
    config_out = {'effective': effective, 'pyradiomics_version': radiomics.__version__,
                  'simpleitk_version': _sitk().Version_VersionString(), 'roi_label': roi_label,
                  'samples': samples.to_dict('records')}
    config_out['parameter_sha256'] = hashlib.sha256(json.dumps(effective, sort_keys=True, default=str).encode()).hexdigest()
    return ExtractionResult(pd.DataFrame([r[0] for r in results]), pd.DataFrame([r[1] for r in results]), config_out)
