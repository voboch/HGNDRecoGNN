"""Read-only single-file detector/primary namespace check; no efficiency claim."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def binding_diagnostics(v, p):
    if p.duplicated(['Row', 'Id']).any():
        raise ValueError('Nonunique primary Row/Id')
    use = v.loc[(v.fMotherId == -1) & v.Side.isin([5, 6])].drop_duplicates(['Row', 'Id'])
    joined = use.merge(p, on=['Row', 'Id'], how='left', indicator=True, suffixes=('_det', '_prim'))
    missing = joined.loc[joined._merge != 'both']
    neutron = joined.PDG_det.abs().to_numpy() == 2112
    agree = (joined._merge == 'both').to_numpy() & (joined.PDG_det.to_numpy() == joined.PDG_prim.to_numpy())
    # Two last-digit units of the six-decimal CSV vertex serialization.
    # This verifies namespace identity; it does not select a physics window.
    for col in ['vX', 'vY', 'vZ']:
        agree &= np.isclose(joined[col+'_det'], joined[col+'_prim'], rtol=0, atol=2e-6)
    def counts(series):
        return {str(int(k)): int(n) for k, n in series.value_counts().items()}
    result = {'scope': 'Single-file engineering binding probe, not detection efficiency',
              'primary_rows': len(p), 'primary_events': int(p.Row.nunique()),
              'vacs_export_events': int(v.Row.nunique()), 'primary_particle_counts': counts(p.PDG),
              'front_face_primary_detector_records': len(use), 'missing_primary_id': len(missing),
              'missing_pdg_counts': counts(missing.PDG_det),
              'missing_rows_outside_primary_events': int((~missing.Row.isin(p.Row.unique())).sum()),
              'missing_neutron_ids': int((missing.PDG_det.abs() == 2112).sum()),
              'neutron_records': int(neutron.sum()),
              'neutron_matched_ids': int((neutron & (joined._merge == 'both').to_numpy()).sum()),
              'neutron_pdg_and_vertex_agree': int((neutron & agree).sum()),
              'source_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    nucleon = joined.PDG_det.abs().isin([2112, 2212])
    found = joined._merge == 'both'
    same_pdg = joined.PDG_det == joined.PDG_prim
    result['nucleon_binding'] = {
        'eligible': int(nucleon.sum()), 'matched_ids': int((nucleon & found).sum()),
        'pdg_agree': int((nucleon & found & same_pdg).sum()),
        'pdg_vertex_agree': int((nucleon & agree).sum()),
        'missing_pdg_counts': counts(joined.loc[nucleon & ~found, 'PDG_det'])}
    # Both exporters print the same CbmMCTrack fields with six decimals.
    # Search the full supplied primary file to distinguish a simple Row
    # reordering from absence of the detector particles. No cut is fitted.
    keys = ['PDG', 'Ekin', 'Px', 'Py', 'Pz']
    candidates = use.loc[use.PDG.abs().isin([2112, 2212])]
    signatures = candidates.merge(p, on=keys, suffixes=('_det', '_prim'))
    vertices = candidates[['Row', 'vX', 'vY', 'vZ']].drop_duplicates().merge(
        p[['Row', 'vX', 'vY', 'vZ']].drop_duplicates(), on=['vX', 'vY', 'vZ'],
        suffixes=('_det', '_prim'))
    result['exact_signature_search'] = {
        'scope': 'same supplied primary file; exact exported numbers; no fitted alignment',
        'detector_nucleons_with_momentum_energy_match': len(signatures[['Row_det','Id_det']].drop_duplicates()),
        'detector_events_with_vertex_match': int(vertices.Row_det.nunique())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vacs', type=Path, required=True)
    parser.add_argument('--primary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    v = pd.read_csv(args.vacs)
    p = pd.read_csv(args.primary, usecols=['Row', 'Id', 'PDG', 'vX', 'vY', 'vZ', 'Ekin', 'Px', 'Py', 'Pz'])
    result = binding_diagnostics(v, p)
    result['inputs'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in [args.vacs, args.primary]}
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
