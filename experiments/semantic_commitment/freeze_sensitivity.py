#!/usr/bin/env python3
"""Deterministically serialize the frozen E4 INTERNAL_PLUS_ACTION ridge score.

No candidate is selected here.  The feature family, training partition, ridge
penalty, log target, and original table are all inherited from E4.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.server_deep_validation.pv0_overnight_common import atomic_write_json

RAW = Path('artifacts/semantic_risk/e4_e5_e6_raw.parquet')
E6 = Path('artifacts/semantic_risk/e6_semantic_risk.parquet')
OUT = Path('reports/semantic_commitment/FROZEN_SENSITIVITY_MODEL.json')


def checksum(payload: dict) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(canonical).hexdigest()


def main() -> None:
    raw = pd.read_parquet(RAW)
    e6 = pd.read_parquet(E6)
    features = sorted(c for c in raw.columns if c.startswith('internal_')) + [
        'action_norm', 'endpoint_displacement', 'action_curvature', 'action_jerk', 'gripper_transition'
    ]
    if len(features) != 89:
        raise RuntimeError(f'expected frozen 89 features, got {len(features)}')
    train = raw.query("split in ['discovery', 'validation']").copy()
    x = train[features].to_numpy(dtype=np.float64)
    y = np.log1p(train['causal_sensitivity_target'].to_numpy(dtype=np.float64))
    mean = x.mean(axis=0); scale = x.std(axis=0); scale[scale < 1e-12] = 1.0
    coef = np.linalg.solve(((x - mean) / scale).T @ ((x - mean) / scale) + 10.0*np.eye(len(features)), ((x - mean) / scale).T @ (y-y.mean()))
    intercept = float(y.mean())
    pred = np.expm1(((x-mean)/scale) @ coef + intercept)
    current = train[['state_id']].copy()
    current['reconstructed_prediction'] = pred
    reference = e6.query("split in ['discovery', 'validation']")[['state_id', 'pred_sensitivity']].copy()
    joined = current.merge(reference, on='state_id', validate='one_to_one')
    if len(joined) != len(train):
        raise RuntimeError('E4 raw/e6 state IDs do not align')
    max_error = float(np.max(np.abs(joined.reconstructed_prediction - joined.pred_sensitivity)))
    if max_error > 1e-10:
        raise RuntimeError(f'frozen E4 reconstruction mismatch: {max_error}')
    model = {
        'schema_version': 1,
        'source_experiment': 'semantic_risk E4 INTERNAL_PLUS_ACTION',
        'selection_history': 'Family selected on original validation; this artifact uses the original post-selection refit on discovery+validation only.',
        'training_splits': ['discovery', 'validation'],
        'training_rows': int(len(train)),
        'target': 'log1p(causal_sensitivity_target)',
        'output_transform': 'expm1(linear_score)',
        'ridge_lambda': 10.0,
        'dtype': 'float64 reference arithmetic',
        'feature_names': features,
        'feature_mean': mean.tolist(),
        'feature_scale': scale.tolist(),
        'coefficients': coef.tolist(),
        'intercept': intercept,
        'score_direction': 'higher predicts higher E4 causal sensitivity target',
        'route_validity': 'P1_ONLY_PENDING_E11A_ROUTE_TRANSFER',
        'reconstruction_validation': {
            'rows': int(len(joined)), 'max_abs_prediction_error_vs_e4_saved_predictions': max_error,
            'tolerance': 1e-10, 'status': 'PASS',
        },
    }
    model['checksum_sha256'] = checksum(model)
    atomic_write_json(OUT, model)
    print(json.dumps({'status':'PASS','features':len(features),'rows':len(train),'max_error':max_error,'checksum':model['checksum_sha256']}))


if __name__ == '__main__':
    main()
