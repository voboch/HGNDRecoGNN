"""Pin slide equations, truth independence and DST bookkeeping failures."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from proton_pid.core import MASS_GEV, bethe_bloch, fit_calibration, select_protons
from proton_pid.root_io import tof_values


def calibration():
    cal = {"schema_version": 1, "pid_method": "parfenov_combined", "tof": "700", "species": {}}
    for name,pdg in (("proton",2212),("pion",211)):
        cal["species"][name] = {"mass_GeV": MASS_GEV[pdg],
            "bethe_parameters": [80,5,1,2,2], "p": [.5,1,2,3,4],
            "sigma_dedx_relative": [.03]*5, "sigma_m2": [.03]*5}
    return cal


def track_frame(n=4):
    p=np.full(n,1.)
    return pd.DataFrame({"source_id":["one"]*n,"entry":np.arange(n),"track_index":np.zeros(n,int),
        "p":p,"dedx":bethe_bloch(p,MASS_GEV[2212],[80,5,1,2,2]),
        "charge_sign":np.ones(n,int),"from_vertex":np.ones(n,int),
        "m2_400":np.full(n,np.nan),"m2_700":np.full(n,MASS_GEV[2212]**2),
        "tof400_valid":np.zeros(n,int),"tof700_valid":np.ones(n,int)})


def test_exact_slide_distances_and_selection_ignore_truth():
    d=track_frame()
    d.loc[1,"dedx"] *= 1+2.1*.03
    d.loc[2,"charge_sign"]=-1
    d.loc[3,"from_vertex"]=0
    d["mc_pdg"]=[211,2212,2212,2212]  # deliberately wrong labels for selection
    first=select_protons(d,calibration())
    assert first.selected_proton.tolist()==[True,False,False,False]
    assert first.loc[1,"x_proton"] == pytest.approx(2.1)
    d["mc_pdg"] = 999
    assert select_protons(d,calibration()).selected_proton.equals(first.selected_proton)
    assert first.loc[0,"ekin_proton_hypothesis"]==pytest.approx(np.hypot(1,MASS_GEV[2212])-MASS_GEV[2212])


def test_unavailable_dedx_tof_and_extrapolation_rejected():
    d=track_frame()
    d.loc[0,"dedx"]=0
    d.loc[1,"tof700_valid"]=0
    d.loc[2,"p"]=5
    d.loc[3,"dedx"]=np.nan
    assert not select_protons(d,calibration()).pid_valid.any()
    d=track_frame(1)
    with pytest.raises(ValueError,match="Duplicate"):
        select_protons(pd.concat([d,d]),calibration())


class Array:
    def __init__(self,values): self.values=values
    def GetEntriesFast(self): return len(self.values)
    def At(self,i): return self.values[i]


def test_tof_requires_matched_index_and_keeps_negative_smeared_mass():
    hit=SimpleNamespace(GetLength=lambda:700.,GetTimeStamp=lambda:23.)
    track=SimpleNamespace(GetTof2HitIndex=lambda:-1,GetMass2=lambda _:0.88)
    assert not tof_values(track,Array([hit]),700)["tof700_valid"]
    track.GetTof2HitIndex=lambda:0
    track.GetMass2=lambda _:-.02
    row=tof_values(track,Array([hit]),700)
    assert row["tof700_valid"]==1
    assert row["m2_700"]==-.02
    assert row["beta_700"]==pytest.approx(700/(29.9792458*23))
    track.GetMass2=lambda _:-1000.
    assert not tof_values(track,Array([hit]),700)["tof700_valid"]


def test_zero_track_events_remain_in_denominator(tmp_path,monkeypatch):
    import proton_pid.root_io as io
    class Tree:
        def GetEntries(self): return 3
        def GetEntry(self,_): return 1
        def GetListOfBranches(self): return []
    root=SimpleNamespace(gROOT=SimpleNamespace(GetVersion=lambda:"synthetic"))
    fakefile=SimpleNamespace(Close=lambda:None)
    monkeypatch.setattr(io,"open_root",lambda *args:(root,fakefile,Tree()))
    monkeypatch.setattr(io,"tree_object",lambda *args:None)
    monkeypatch.setattr(io,"mc_array",lambda *args:Array([]))
    p=tmp_path/"fixture.root"; p.write_bytes(b"fixture")
    meta=io.extract(p,tmp_path/"out",require_mc=True)
    events=pd.read_csv(tmp_path/"out_events.csv.gz")
    assert len(events)==3
    assert events.n_global.sum()==0
    # Row is the ZERO-BASED TTree entry, matching Convert2Csv.C, which writes
    # Row = ev from a loop starting at 0. This previously asserted [1,2,3],
    # encoding an entry+1 convention that joined every event's tracks to the
    # NEXT event's HGND hits.
    assert events.Row.tolist()==[0,1,2]
    assert events.Row.tolist()==events.entry.tolist(), \
        "Row must equal entry; any offset breaks the (source_id, Row) join"
    assert meta["completed"] and meta["entries_processed"]==3
    with pytest.raises(FileExistsError): io.extract(p,tmp_path/"out")


def test_mc_template_fit_recovers_response_widths():
    rng=np.random.default_rng(22)
    edges=np.linspace(.5,4.5,9)
    frames=[]
    for pdg in (2212,211):
        p=rng.uniform(.5,4.5,6000)
        mu=bethe_bloch(p,MASS_GEV[pdg],[80,5,1,2,2])
        d=track_frame(len(p))
        d["source_id"]=str(pdg)
        d["p"]=p
        d["dedx"]=mu*(1+rng.normal(0,.04,len(p)))
        d["m2_700"]=MASS_GEV[pdg]**2+rng.normal(0,.05,len(p))
        d["mc_pdg"]=pdg
        frames.append(d)
    d=pd.concat(frames,ignore_index=True)
    cal=fit_calibration(d,700,edges,100,"synthetic development sample")
    assert np.mean(cal["species"]["proton"]["sigma_dedx_relative"])==pytest.approx(.04,abs=.006)
    assert np.mean(cal["species"]["proton"]["sigma_m2"])==pytest.approx(.05,abs=.006)
    assert cal["tof"]=="700"
    json.dumps(cal,allow_nan=False)
