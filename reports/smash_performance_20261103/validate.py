"""Check the accepted report export; no model calls, fits, or unit-test suite."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pymupdf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


pins = json.loads((HERE / "input_pins.json").read_text())["files"]
for path, digest in pins.items():
    assert sha(ROOT / path) == digest, path
data = json.loads((HERE / "data.json").read_text())
assert data["source_sha256"] == pins
assert data["files_verified"] == len(data["csv_hashes"]) == 5112
assert sha(HERE / "counts.npz") == data["counts_sha256"]
z = np.load(HERE / "counts.npz")
analysis = json.loads((ROOT / "results/smash_train_20261028/analysis/summary.json").read_text())
for arm in ("A1", "B0", "B1"):
    for seed in ("42", "123", "456"):
        get = lambda key: z[f"{arm}__{seed}__{key}"]
        assert get("n").sum() == 3572710 and get("n_pos").sum() == 640277
        for field, name in (("T", "n_pos"), ("P", "n_pos"), ("C2", "n_selpos"), ("C3", "n_selpos"), ("C4", "n_sel")):
            assert np.array_equal(get(field).sum(axis=1), get(name)), (arm, seed, field)
        for fold in range(3):
            ix = np.array(data["fold"]) == fold
            old = analysis["arms"][arm][seed]["classification"]["pooled"][f"fold_{fold}"]["all"]
            for key, name in (("n", "n"), ("signal", "n_pos"), ("selected", "n_sel"), ("selected_signal", "n_selpos")):
                assert old[key] == int(get(name)[ix].sum())
for seed in ("42", "123", "456"):
    for key in ("n", "n_pos", "n_sel", "n_selpos", "TP", "FP"):
        assert np.array_equal(z[f"B0__{seed}__{key}"], z[f"B1__{seed}__{key}"])

primary = np.load(HERE / "primary_spectra.npz")
accepted = np.load(ROOT / "results/truth_reach_20261026/aggregate_4453896_verified/result/spectra.npz")
assert np.array_equal(primary["edges"], accepted["e_edges"])
assert .3 in primary["edges"]
for tag in ("zeroSpot", "defaultSpot", "bigSpot"):
    assert np.allclose(primary[tag+"__central"], accepted[tag+"_per_event"], rtol=0, atol=1e-14)
    assert np.allclose(primary[tag+"__replicas"].std(axis=0, ddof=1), accepted[tag+"_sd"], rtol=0, atol=1e-14)

manifest = json.loads((HERE / "figure_manifest.json").read_text())
assert len(manifest) == 23
bounds = json.loads((HERE / "figure_bounds.json").read_text())
assert all(not row["vertical_truncation"] for row in bounds)
for row in bounds:
    if row["ratio_data_bounds"]:
        lo, hi = row["ratio_data_bounds"]
        assert row["ylim"][0] <= lo <= hi <= row["ylim"][1]

pdf = ROOT / "output/pdf/hgnd_smash_performance_20261103.pdf"
doc = pymupdf.open(pdf)
assert len(doc) == 29
fonts = {}
for i, page in enumerate(doc):
    assert (HERE / "render" / f"page-{i+1:02d}.png").exists()
    assert page.get_text().strip()
    for b in page.get_text("dict")["blocks"]:
        if b["type"] != 0:
            continue
        for line in b["lines"]:
            for span in line["spans"]:
                x0, y0, x1, y1 = span["bbox"]
                assert x0 >= -1 and y0 >= -1 and x1 <= page.rect.width+1 and y1 <= page.rect.height+1, (i, span)
                assert "\ufffd" not in span["text"]
    for font in page.get_fonts(full=True):
        assert font[1] == "ttf" and font[2] != "Type3", font
        assert doc.extract_font(font[0])[3], font
        fonts[font[3]] = font[2]
    assert not page.get_images(), "Report figures must remain vectors"

files = [p for p in HERE.rglob("*") if p.is_file() and "render" not in p.parts and p.name != "validation.json" and "__pycache__" not in p.parts]
files += [pdf, HERE.parent / "smash_performance_20261103.md", ROOT / "scripts/plot_b_full.py"]
result = {
    "scope": "accepted saved outputs; no inference, new fits, jobs or unit tests",
    "input_pins_verified": len(pins), "csvs_verified_during_collect": 5112,
    "count_parity": "all per-file integrals and accepted per-fold counts exact",
    "B0_B1_classification_parity": True,
    "primary_saved_central_and_400_replica_SD_replay": True,
    "figure_families": len(manifest), "pages": len(doc),
    "plot_data_vertices_checked": sum(row["data_vertices_checked"] for row in bounds),
    "plot_bounds": "all in-view data vertices/error ends and ratio bands contained",
    "PDF_text_bounds": "all contained", "PDF_fonts": fonts, "PDF_all_fonts_embedded_TrueType": True,
    "PDF_vector_figures": 23,
    "render": "Poppler, 110 dpi; all 29 pages visually inspected",
    "visual_review": {"date": "2026-11-03", "pdf_sha256": sha(pdf), "pages": list(range(1, len(doc)+1)), "figures": [v["name"] for v in manifest]},
    "files_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)},
}
(HERE / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
print(f"Validated {len(pins)} inputs, {len(manifest)} figure families, {len(doc)} PDF pages")
