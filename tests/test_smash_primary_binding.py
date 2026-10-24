"""Separate a recoverable Row namespace shift from different MC particles."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
from HGNDRecoGNN.scripts.check_smash_primary_binding import binding_diagnostics


def inputs():
    v = pd.DataFrame(dict(Row=[0,0], Id=[5,6], PDG=[2112,2212], fMotherId=[-1,-1],
         Side=[5,5], Ekin=[1.,2.], Px=[.1,.2], Py=[.3,.4], Pz=[1.5,2.5],
         vX=[0.,0.], vY=[0.,0.], vZ=[0.,0.]))
    return v, v.drop(columns=['fMotherId','Side']).copy()


def test_exact_signatures_expose_row_reordering_without_inventing_a_mapping():
    v, p = inputs()
    p.Row = 10
    r = binding_diagnostics(v, p)
    assert r['nucleon_binding']['matched_ids'] == 0
    assert r['exact_signature_search']['detector_nucleons_with_momentum_energy_match'] == 2


def test_common_event_vertex_does_not_validate_wrong_particle_ids():
    v, p = inputs()
    p.PDG = [2212,2112]
    r = binding_diagnostics(v, p)
    assert r['nucleon_binding']['matched_ids'] == 2
    assert r['nucleon_binding']['pdg_agree'] == 0
    assert r['nucleon_binding']['pdg_vertex_agree'] == 0
    assert r['exact_signature_search']['detector_nucleons_with_momentum_energy_match'] == 0
