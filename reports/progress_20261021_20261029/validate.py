"""Independent arithmetic/source and PDF checks for the progress artifact."""

from pathlib import Path
import hashlib
import json
import math
import os
import re
import statistics
import subprocess

import pdfplumber
from pypdf import PdfReader

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
D=json.loads((HERE/'data.json').read_text())
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks=0
for path,expected in D['sources_sha256'].items():
    assert h(REPO/path)==expected,path
    checks+=1
arms=D['training']['classification_energy']['arms']
for arm in arms.values():
    for value in arm.values():
        if isinstance(value,dict) and 'per_seed' in value:
            a=value['per_seed']
            assert len(a)==3
            assert math.isclose(statistics.mean(a),value['mean'],abs_tol=1e-12)
            assert math.isclose(statistics.stdev(a),value['seed_sd'],abs_tol=1e-12)
            checks+=2
for key,value in D['training']['classification_energy']['B1_minus_A1'].items():
    delta=[b-a for a,b in zip(arms['A1'][key]['per_seed'],arms['B1'][key]['per_seed'])]
    assert all(math.isclose(a,b,abs_tol=1e-12) for a,b in zip(delta,value['per_seed']))
    assert math.isclose(statistics.mean(delta),value['mean'],abs_tol=1e-12)
    assert math.isclose(statistics.stdev(delta)/math.sqrt(3),value['se_of_mean'],abs_tol=1e-12)
    checks+=3
pooled=D['training']['rn_ladder_primary']['pooled']
for arm,value in pooled['arms'].items():
    gap=[x-value['T1_percent'] for x in value['C4_percent']]
    assert all(math.isclose(a,b,abs_tol=1e-12) for a,b in zip(gap,value['steps_percent']['C4-T1']))
    checks+=1
delta=[b-a for a,b in zip(pooled['arms']['A1']['C4_percent'],pooled['arms']['B1']['C4_percent'])]
assert all(math.isclose(a,b,abs_tol=1e-12) for a,b in zip(delta,pooled['B1_minus_A1_C4']['per_seed']))
assert all(v['accepted'] and v['test_files']==v['scored_files'] for v in D['evaluation_acceptance'].values())
assert sum(v['test_files'] for v in D['evaluation_acceptance'].values())==568
assert len(D['training_check']['tasks'])==9 and all(t['ok'] for t in D['training_check']['tasks'].values())
assert D['purity_preflight']['status'].startswith('NO-GO')
checks+=5

pdf=HERE/'HGND_progress_20261021_20261029.pdf'
reader=PdfReader(pdf)
assert len(reader.pages)==13
word_bounds=[]
with pdfplumber.open(pdf) as doc:
    for i,page in enumerate(doc.pages):
        words=page.extract_words()
        assert len(words)>40,(i+1,len(words))
        bad=[w for w in words if w['x0']<30 or w['x1']>page.width-30 or w['top']<12 or w['bottom']>page.height-12]
        assert not bad,(i+1,bad[:2])
        word_bounds.append({'page':i+1,'words':len(words),'out_of_bounds':0})
font_tool='/Users/vovvy/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/poppler/bin/pdffonts'
fontconfig=Path('/private/tmp/hgnd_report_fontconfig.conf')
fontconfig.write_text('<?xml version="1.0"?><fontconfig><dir>/System/Library/Fonts</dir><cachedir>/private/tmp/hgnd-report-fontcache</cachedir></fontconfig>')
result=subprocess.run([font_tool,str(pdf)],capture_output=True,text=True,check=True,
    env={**os.environ,'FONTCONFIG_FILE':str(fontconfig),'XDG_CACHE_HOME':'/private/tmp/hgnd-cache'})
listing=result.stdout
assert not result.stderr,result.stderr[:500]
assert 'Type 3' not in listing and not re.search(r'\sno\s',listing)
images=0
def count_images(resources):
    count=0
    for x in resources.get('/XObject',{}).values():
        obj=x.get_object()
        if obj.get('/Subtype')=='/Image':count+=1
        if obj.get('/Subtype')=='/Form' and '/Resources' in obj:count+=count_images(obj['/Resources'])
    return count
for page in reader.pages:images+=count_images(page['/Resources'])
assert images==0,'Figures must remain vector content'

fig_hashes=json.loads((HERE/'figure_hashes.json').read_text())
assert len(fig_hashes)==18
for name,expected in fig_hashes.items():assert h(HERE/'figures'/name)==expected,name
record={'status':'arithmetic_sources_fonts_and_bounds_passed','date':'2026-10-29','material_cutoff':D['document_cutoff'],'source_head':D['source_head'],'source_files_checked':len(D['sources_sha256']),'arithmetic_checks':checks,'pdf_sha256':h(pdf),'markdown_sha256':h(HERE.parent/'HGND_progress_20261021_20261029.md'),'data_sha256':h(HERE/'data.json'),'status_snapshot_sha256':h(HERE/'slurm_status.json'),'script_sha256':{p:h(HERE/p) for p in ['build.py','plot.py','validate.py']},'figure_count':9,'figure_sha256':fig_hashes,'pdf_pages':13,'word_bounds':word_bounds,'all_fonts_embedded':True,'type3_fonts':0,'raster_images_in_pdf':images,'visual_inspection':'pending: must be added after viewing the latest renders','scope':'Document generation and arithmetic only; no raw-data reduction, inference, training or new physics experiment'}
(HERE/'validation.json').write_text(json.dumps(record,indent=2)+'\n')
print('Passed:',checks,'source/arithmetic checks; 13 pages; all text inside page; embedded fonts; 9 vector figures.')
