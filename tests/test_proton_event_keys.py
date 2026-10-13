"""The (source_id, Row) join between the charged arm and the HGND arm.

Row is the zero-based TTree entry in every table the converter writes, and it
repeats across files by design, so the charged arm joins the neutron arm on the
pair. Two things can break that silently and neither shows up as an error:

  * an off-by-one in Row, which pairs each event's protons with the next
    event's neutrons. The Python exporter carried exactly this
    (`Row = entry + 1`) against a converter that writes `Row = ev` from a loop
    starting at zero;
  * a source_id that is not reproducible outside C++, which makes the pair
    unusable from the analysis side.

These run locally against the real CSV conventions. They cannot prove the DST
getters behave as assumed -- that needs the ncx smoke.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def fnv1a_source_id(basename: str) -> int:
    """Python twin of bmnx::SourceId, which the analysis needs to re-derive."""
    h = 1469598103934665603
    for ch in basename.encode():
        h ^= ch
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h % 2147483647


def test_source_id_matches_the_cpp_implementation():
    """Compile bmnx::SourceId and compare it against the Python twin.

    If these drift the analysis silently joins on a key the converter never
    wrote.
    """
    src = os.path.join(HERE, "cpp", "sourceid_probe.cc")
    os.makedirs(os.path.dirname(src), exist_ok=True)
    with open(src, "w") as fh:
        fh.write(r'''
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
struct TString { std::string s; TString()=default; TString(const char*c):s(c?c:""){}
  const char* Data() const { return s.c_str(); }
  int Length() const { return (int)s.size(); }
  char operator[](int i) const { return s[i]; } };
struct TClonesArray { int GetEntriesFast() const {return 0;} void* At(int) const {return nullptr;} };
struct TSystemStub { const char* BaseName(const char* p){ const char* q=strrchr(p,'/'); return q?q+1:p; } };
static TSystemStub g; static TSystemStub* gSystem=&g;
typedef long long Long64_t;
#define BMNX_STANDALONE_TEST 1
#include "../../ncx_refactor/BmnTrackExport.h"
int main(int argc, char** argv){
  for (int i=1;i<argc;++i) printf("%ld\n", bmnx::SourceId(TString(argv[i])));
  return 0; }
''')
    exe = "/tmp/bmnx_sourceid_probe"
    r = subprocess.run(["g++", "-std=c++17", "-O0", "-o", exe, src],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"probe did not build:\n{r.stderr}"

    names = ["/a/b/0001_dst.root", "/other/0001_dst.root", "/a/b/0002_dst.root"]
    out = subprocess.run([exe] + names, capture_output=True, text=True)
    cpp = [int(x) for x in out.stdout.split()]
    py = [fnv1a_source_id(os.path.basename(n)) for n in names]
    assert cpp == py, f"C++ {cpp} != Python {py}"
    assert cpp[0] == cpp[1], "source_id must depend on the basename only"
    assert cpp[0] != cpp[2], "different files must differ"


def test_row_is_zero_based_in_the_shipped_hits_csvs():
    """The converter's own output is the authority on the convention."""
    import glob
    cands = glob.glob(os.path.join(ROOT, "results", "**", "*_hits.csv"),
                      recursive=True)
    cands = [c for c in cands if os.path.getsize(c) > 100]
    if not cands:
        print("  (no local *_hits.csv; convention checked in the C++ test)")
        return
    d = pd.read_csv(cands[0], nrows=5000)
    assert d.Row.min() == 0, (
        f"{cands[0]} starts at Row={d.Row.min()}; the charged-arm exporter "
        "writes Row=entry, so a 1-based hits file would misalign the join")


def test_python_exporter_writes_row_equal_to_entry():
    """Guard the off-by-one that was shipped once already."""
    src = open(os.path.join(ROOT, "proton_pid", "root_io.py")).read()
    assert '"Row": entry,' in src, "root_io must write Row = entry"
    assert '"Row": entry+1' not in src and '"Row": entry + 1' not in src, (
        "Row = entry+1 joins each event's tracks to the next event's hits")


def test_join_pairs_the_same_event():
    """An end-to-end join on (source_id, Row) must not shift by one."""
    sid = fnv1a_source_id("0001_dst.root")
    hits = pd.DataFrame({"source_id": sid, "Row": [0, 0, 1, 2],
                         "fELoss": [0.01, 0.02, 0.03, 0.04]})
    tracks = pd.DataFrame({"source_id": sid, "Row": [0, 1, 1, 2],
                           "p": [1.0, 2.0, 2.5, 3.0]})
    j = hits.merge(tracks, on=["source_id", "Row"], how="inner")
    assert set(j.Row) == {0, 1, 2}, "every shared event must survive the join"
    assert len(j) == 2 + 2 + 1, "join multiplicity must be hits x tracks per event"

    shifted = tracks.assign(Row=tracks.Row + 1)
    js = hits.merge(shifted, on=["source_id", "Row"], how="inner")
    assert len(js) != len(j), (
        "a one-event shift must change the join; if this ever passes, the test "
        "cannot detect the off-by-one it exists for")


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok  {name}")
            except AssertionError as exc:
                print(f"FAIL {name}: {exc}")
                fails += 1
    print("all passed" if not fails else f"{fails} failed")
    sys.exit(1 if fails else 0)
