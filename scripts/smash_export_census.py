"""Exact content census of every corrected-SMASH detector export, per potential.

Read-only bookkeeping on raw exports: no model, fitted statistic, Edep cut or
time jitter enters. Identity is the raw exported content at three levels:
SHA-256 of both files, raw pre-jitter hit event signatures over HIT_FIELDS
(content_census.signature), and hashed direct-primary MCTrack records over the
exact mapper keys (map_smash_primary.KEYS). Every byte-unique content is bound
to the tag's primary files with the exact mapper: exact PDG, Ekin, momentum and
vertex, the same Row/Id, one source file, ambiguity refused.

Contents sharing bytes, an event signature, a primary file, or at least two
direct-primary records in one event form one group. A group keeps the content
with the most events (ties: lexicographically first member name, which keeps
the pinned bound member and its path-seeded audit jitter). Every dropped
content must be an event-level subset of the kept one, or the census stops.

Exit status: 0 complete, 3 stopped on a census condition (evidence written and
exported), anything else a crash (nothing exported).
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import errno
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from HGNDRecoGNN.scripts.audit_smash_sources import HITS, VACS  # noqa: E402
from HGNDRecoGNN.scripts.bfree_truth_campaign import runtime, source_digest  # noqa: E402
from HGNDRecoGNN.scripts.build_smash_bound_inventory import EMPTY_BYTES, EMPTY_HASH  # noqa: E402
from HGNDRecoGNN.scripts.content_census import (  # noqa: E402
    HIT_FIELDS,
    event_signatures,
    sha256,
    write_index,
)
from HGNDRecoGNN.scripts.map_smash_primary import (  # noqa: E402
    COLS,
    KEYS,
    binding_candidates,
    targets_from_vacs,
)

SCHEMA = "smash-export-census-v1"
TAGS = ("zeroSpot", "defaultSpot", "bigSpot")
INVENTORY = "bound_inventory.json"
# Non-alias, non-empty pinned bindings of the frozen 2026-10-23 inventory:
# 195 + 186 + 51 = 432. A different count means a different inventory.
PINNED_BOUND = {"zeroSpot": 195, "defaultSpot": 186, "bigSpot": 51}
# Two shared direct-primary records in one event pair is the smallest count a
# single coincident six-decimal record cannot produce. Not optimised: the
# per-pair histogram of shared records is written to the census so a reader
# can see what one or three would have merged instead.
MIN_SHARED_RECORDS = 2
# Same chunk as the mapper: bounds parsing memory, not a physics selection.
PRIMARY_CHUNK = 100_000
RECORD_PREFIX = ("primary-record-v1:" + ",".join(KEYS)).encode()
# Smoke: per tag a pinned bound member, a byte alias of it (SHA-verified on
# the login node 2026-10-28) and a pinned or listed header-only export.
SMOKE_MEMBERS = {
    "zeroSpot": ("10943245/0001", "10943245/0629", "10943245/0196"),
    "defaultSpot": ("10943245/0008", "10943245/0395", "10943245/0187"),
    "bigSpot": ("10943245/0008", "10943245/0031", "10943245/0002"),
}
# One partition of the test queue is 30 min; the full array asks 1:30:00.
FULL_LIMIT_SECONDS = 5400
# Projection margin: 1.5x plus 5 min covers cold /scratch reads and the
# per-chunk merge growing with the full target table; it is a scheduling
# allowance, not a measured bound.
MARGIN_FACTOR, MARGIN_SECONDS = 1.5, 300
STOPPED = 3
# RA-09: on compute nodes, opening a readable /scratch input has failed with
# EACCES at a different file each time (full census 4464315 and retry 4464423,
# cn-036..039, 2026-10-28); the same files read on login and on other nodes.
# Every input access is retried after these waits, and each retry is written
# to the summary and the job log. The schedule is a convention, not a
# measurement: no outage duration has been observed. The worst case adds 63 s
# per file against the 1:30:00 limit; a longer outage still crashes the task
# with nothing exported. A retried read cannot change a result unnoticed:
# every input is hashed, pinned hashes are checked and each parsed content is
# hashed again.
ACCESS_RETRY_WAITS = (1, 2, 4, 8, 16, 32)
ACCESS_RETRIES = []


class CensusStop(Exception):
    """A census condition (hash, membership, binding, subset) failed."""


def with_access_retry(call, path, *args, **kwargs):
    """call(path, ...) retried on EACCES only; other errors pass through."""
    for attempt, wait in enumerate((*ACCESS_RETRY_WAITS, None), start=1):
        try:
            return call(path, *args, **kwargs)
        except PermissionError as error:
            if wait is None or error.errno != errno.EACCES:
                raise
            event = {"path": str(path), "call": getattr(call, "__name__", repr(call)),
                     "attempt": attempt, "wait_seconds": wait, "node": os.uname().nodename,
                     "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            ACCESS_RETRIES.append(event)
            print("access retry " + json.dumps(event), file=sys.stderr, flush=True)
            time.sleep(wait)


def read_sha256(path):
    return with_access_retry(sha256, path)


def stat_size(path):
    return with_access_retry(os.stat, path).st_size


def list_dir(path):
    with os.scandir(path) as entries:
        return [(e.name, e.is_dir()) for e in entries]


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def read_export(path, schema):
    frame = with_access_retry(pd.read_csv, path)
    if list(frame.columns) != list(schema):
        raise CensusStop(f"Unexpected CSV schema: {path}")
    return frame


def record_hashes(targets):
    """Hash each direct-primary record over the mapper keys, Row/Id excluded.

    Row and Id are left out so that a renumbered copy still collides.
    """
    a = np.array(targets[KEYS].to_numpy(dtype="<f8"), dtype="<f8", order="C", copy=True)
    a[a == 0] = 0  # Equal physical zero irrespective of sign bit.
    return [hashlib.sha256(RECORD_PREFIX + row.tobytes()).hexdigest() for row in a]


def multiset_digest(values):
    h = hashlib.sha256(b"multiset-v1\n")
    for value in sorted(values):
        h.update(value.encode() + b"\n")
    return h.hexdigest()


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Deterministic root: the smaller name.
            ra, rb = sorted((ra, rb))
            self.parent[rb] = ra
        return ra != rb


def pair_names(raw_root):
    """Export pairs one directory below raw_root, as glob("*/*_hits.csv") finds them.

    Listed explicitly because Path.glob skips a directory it cannot open, which
    under RA-09 would turn an access failure into a membership change.
    """
    hits, vacs = set(), set()
    for sub, is_dir in with_access_retry(list_dir, raw_root):
        if not is_dir:
            continue
        for name, _ in with_access_retry(list_dir, raw_root / sub):
            for suffix, found in (("_hits.csv", hits), ("_vacs.csv", vacs)):
                if name.endswith(suffix):
                    found.add(f"{sub}/{name.removesuffix(suffix)}")
    if hits != vacs:
        raise CensusStop("Unpaired hits/vacs exports: " + ", ".join(sorted(hits ^ vacs)[:5]))
    return sorted(hits)


def hash_files(paths, workers):
    """SHA-256 of many files; hashlib releases the GIL on large reads."""
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return dict(zip(paths, pool.map(read_sha256, paths)))


def scan_primaries(targets, primaries, pinned_sha):
    """The mapper's exact join of `targets` against each primary file.

    A chunk row can only join if its Ekin equals some target Ekin, so rows are
    pre-filtered on that one key before the full eight-key merge; the matches
    are identical to merging the whole chunk and the cost stops growing with
    the target table.
    """
    ekin = np.unique(targets.Ekin.to_numpy(dtype=float))
    matches, files = [], []
    for member, path in primaries:
        started, count = time.monotonic(), 0
        for chunk in with_access_retry(pd.read_csv, path, usecols=COLS, chunksize=PRIMARY_CHUNK):
            if not np.isfinite(chunk.to_numpy(dtype=float)).all():
                raise CensusStop("Nonfinite primary inventory attributes in " + member)
            chunk = chunk.loc[np.isin(chunk.Ekin.to_numpy(dtype=float), ekin)]
            joined = targets.merge(chunk, on=KEYS, suffixes=("_det", "_prim"))
            if len(joined):
                joined["primary_member"] = member
                matches.append(joined)
                count += len(joined)
        digest = read_sha256(path)
        if pinned_sha.get(member) is not None and digest != pinned_sha[member]:
            raise CensusStop("Primary file differs from the pinned inventory: " + member)
        files.append({"member": member, "sha256": digest, "bytes": stat_size(path),
                      "matches": count, "seconds": time.monotonic() - started})
    columns = ["Row_det", "Id_det", *KEYS, "detector_member", "Row_prim", "Id_prim", "primary_member"]
    found = pd.concat(matches, ignore_index=True) if matches else pd.DataFrame(columns=columns)
    return found, files


def primary_links(matches, binding):
    """(content, primary) pairs: exact bindings and >= 2 matched records in one event."""
    links = {(c, b["primary"]) for c, b in binding.items() if b["primary"] is not None}
    if len(matches):
        per_event = (matches.drop_duplicates(["detector_member", "Row_det", "Id_det", "primary_member"])
                     .groupby(["detector_member", "Row_det", "primary_member"]).size())
        for (content, _, primary), n in per_event.items():
            if n >= MIN_SHARED_RECORDS:
                links.add((content, primary))
    return links


def shared_record_pairs(records, key="content"):
    """Event pairs of different contents sharing direct-primary record hashes."""
    a = records.rename(columns={key: "a", "Row": "row_a"})
    b = records.rename(columns={key: "b", "Row": "row_b"})
    m = a.merge(b, on="h")
    m = m.loc[m.a < m.b]
    if not len(m):
        return pd.Series(dtype=int)
    return m.groupby(["a", "row_a", "b", "row_b"]).size()


def contained(dropped, kept, raw_root):
    """Event-level subset test of one content in another.

    Signatures must form a sub-multiset. The final event of a truncated copy
    may itself be cut; it passes only if its raw hit rows are a sub-multiset of
    the kept content's rows with the same Row. Direct-primary records must be a
    sub-multiset without exception, since a truncated vacs keeps whole tracks.
    """
    left = Counter(dropped["signatures"]) - Counter(kept["signatures"])
    relation = "event_subset"
    if left:
        final = dropped["final_signature"]
        if sum(left.values()) != 1 or left.get(final) != 1:
            return None, {"missing_events": sum(left.values())}
        rows = []
        for content in (dropped, kept):
            frame = read_export(raw_root / (content["id"] + "_hits.csv"), HITS)
            part = frame.loc[frame.Row == dropped["final_row"], list(HIT_FIELDS)].to_numpy(dtype="<f8")
            part[part == 0] = 0
            rows.append(Counter(map(tuple, part)))
        if rows[0] - rows[1]:
            return None, {"missing_events": 1, "final_event_rows_not_contained": True}
        relation = "event_subset_partial_final_event"
    missing = Counter(dropped["records"]) - Counter(kept["records"])
    if missing:
        return None, {"missing_records": sum(missing.values())}
    return relation, {}


def vertex_spread(targets):
    """Descriptive MC-truth spread of direct-primary production vertices."""
    if not len(targets):
        return None
    q = [0.0, 0.01, 0.16, 0.5, 0.84, 0.99, 1.0]
    out = {"unit": "cm (exported vX/vY/vZ)", "quantiles": q, "records": int(len(targets))}
    events = targets.groupby(["detector_member", "Row"])[["vX", "vY", "vZ"]].mean()
    out["events"] = int(len(events))
    for name, frame in (("per_record", targets[["vX", "vY", "vZ"]]), ("per_event_mean", events)):
        out[name] = {c: {"mean": float(frame[c].mean()),
                         "sd": float(frame[c].std(ddof=1)) if len(frame) > 1 else 0.0,
                         "q": [float(x) for x in frame[c].quantile(q)]} for c in ("vX", "vY", "vZ")}
    return out


def census_tag(info, tag, output, *, members=None, primaries=None, workers=1, meta=None,
               expected_pinned_bound=None):
    """Census one potential. Returns the summary; never raises CensusStop."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    raw_root, primary_root = Path(info["raw_root"]), Path(info["primary_root"])
    pinned = info["input_sha256"]
    pinned_primary = info["primary_member_map"]
    aliases = info["duplicate_exports"]
    summary = {"schema": SCHEMA, "tag": tag, "status": "running", "stop_reasons": [],
               "coverage": "full" if members is None else "subset",
               "raw_root": str(raw_root), "primary_root": str(primary_root), **(meta or {}),
               "timings": {}}
    ACCESS_RETRIES.clear()
    try:
        _census(info, tag, output, raw_root, primary_root, pinned, pinned_primary, aliases,
                members, primaries, workers, summary, expected_pinned_bound)
    except (CensusStop, ValueError) as error:
        summary["stop_reasons"].append({"kind": "condition", "detail": str(error)})
    summary["access_retries"] = list(ACCESS_RETRIES)
    summary["status"] = "stopped" if summary["stop_reasons"] else "complete"
    write_json(output / "summary.json", summary)
    return summary


def _census(info, tag, output, raw_root, primary_root, pinned, pinned_primary, aliases,
            members, primaries, workers, summary, expected_pinned_bound):
    t0 = time.monotonic()
    names = pair_names(raw_root)
    if names != sorted(info["all_pair_names"]):
        raise CensusStop("Detector export membership differs from the frozen inventory")
    if members is not None:
        if not set(members) <= set(names):
            raise CensusStop("Smoke member not in the export membership")
        names = sorted(members)
    paths = {n: (raw_root / (n + "_hits.csv"), raw_root / (n + "_vacs.csv")) for n in names}
    digests = hash_files([p for pair in paths.values() for p in pair], workers)
    files = {}
    for n, (hp, vp) in paths.items():
        files[n] = {"hits": digests[hp], "vacs": digests[vp],
                    "hits_bytes": stat_size(hp), "vacs_bytes": stat_size(vp)}
    summary["timings"]["hash"] = {"seconds": time.monotonic() - t0, "files": 2 * len(files),
                                  "bytes": sum(f["hits_bytes"] + f["vacs_bytes"] for f in files.values())}

    # Pinned inventory: every frozen hash, header-only and alias in scope.
    checked = [n for n in names if n in pinned]
    for n in checked:
        if {"hits": files[n]["hits"], "vacs": files[n]["vacs"]} != pinned[n]:
            raise CensusStop("Detector export differs from its pinned hash: " + n)
    pinned_bound = sorted(n for n in checked if pinned_primary.get(n) is not None and n not in aliases)
    if expected_pinned_bound is not None and len(pinned_bound) != expected_pinned_bound:
        raise CensusStop(f"Pinned bound files {len(pinned_bound)} != {expected_pinned_bound}")
    summary["pinned"] = {"hashes_verified": len(checked), "bound_members_verified": len(pinned_bound),
                         "header_only_declared": sum(n in info["empty_exports"] for n in checked),
                         "aliases_declared": sum(n in aliases for n in checked)}

    # Header-only exports: exact pinned header bytes and zero parsed rows.
    header_only = []
    for n in names:
        f = files[n]
        small = f["hits_bytes"] == EMPTY_BYTES["hits"] and f["vacs_bytes"] == EMPTY_BYTES["vacs"]
        if not small:
            if n in info["empty_exports"]:
                raise CensusStop("Declared empty export contains rows: " + n)
            continue
        rows = [len(read_export(p, s)) for p, s in zip(paths[n], (HITS, VACS))]
        if rows != [0, 0] or {"hits": f["hits"], "vacs": f["vacs"]} != EMPTY_HASH:
            raise CensusStop("Header-sized export is not the pinned header-only content: " + n)
        header_only.append(n)
    summary["membership"] = {"pairs": len(names), "header_only": len(header_only),
                             "nonempty": len(names) - len(header_only)}

    # Byte contents: identical (hits, vacs) pairs; id = first member name.
    by_pair = defaultdict(list)
    for n in names:
        if n not in header_only:
            by_pair[(files[n]["hits"], files[n]["vacs"])].append(n)
    contents = {}
    for (hits_sha, vacs_sha), group in by_pair.items():
        group = sorted(group)
        contents[group[0]] = {"id": group[0], "members": group, "hits_sha256": hits_sha,
                              "vacs_sha256": vacs_sha, "hits_bytes": files[group[0]]["hits_bytes"],
                              "vacs_bytes": files[group[0]]["vacs_bytes"]}
    for alias, canonical in aliases.items():
        if alias in names and canonical in names and not any(
                alias in c["members"] and canonical in c["members"] for c in contents.values()):
            raise CensusStop(f"Pinned alias {alias} is not byte-identical to {canonical}")

    # Raw identities per content.
    t1 = time.monotonic()
    events_index, records, targets = [], [], []
    for cid in sorted(contents):
        c = contents[cid]
        hp, vp = paths[cid]
        hits, vacs = read_export(hp, HITS), read_export(vp, VACS)
        if hits.empty or vacs.empty:
            raise CensusStop("Non-header-only export with an empty side: " + cid)
        sigs = list(event_signatures(hits, HIT_FIELDS))
        target = targets_from_vacs(vacs[[*COLS, "fMotherId", "Side"]], cid)
        hashes = record_hashes(target)
        final_row = int(hits.Row.iloc[-1])
        c.update(events=len(sigs), hit_records=int(len(hits)), truth_records=int(len(vacs)),
                 direct_primary_records=int(len(target)), final_row=final_row,
                 row_order_monotone=bool((np.diff(hits.Row.to_numpy()) >= 0).all()),
                 repeated_events_within=sum(v - 1 for v in Counter(s for _, _, s in sigs).values()),
                 signature_digest=multiset_digest([s for _, _, s in sigs]))
        c["signatures"] = [s for _, _, s in sigs]
        c["final_signature"] = next(s for r, _, s in sigs if r == final_row)
        c["records"] = hashes
        events_index.extend((tag, cid, row, n, s) for row, n, s in sigs)
        records.append(pd.DataFrame({"content": cid, "Row": target.Row.to_numpy(np.int64),
                                     "Id": target.Id.to_numpy(np.int64), "h": hashes}))
        targets.append(target)
        if (read_sha256(hp), read_sha256(vp)) != (c["hits_sha256"], c["vacs_sha256"]):
            raise CensusStop("Input changed during census: " + cid)
    summary["timings"]["contents"] = {"seconds": time.monotonic() - t1, "contents": len(contents)}
    records = (pd.concat(records, ignore_index=True) if records
               else pd.DataFrame(columns=["content", "Row", "Id", "h"]))
    targets = pd.concat(targets, ignore_index=True) if targets else None

    # Exact binding of every content to the tag's primary files.
    t2 = time.monotonic()
    primary_names = sorted(info["common_primary_pair_names"]) if primaries is None else sorted(primaries)
    if not set(primary_names) <= set(info["common_primary_pair_names"]):
        raise CensusStop("Primary file outside the frozen primary membership")
    binding, matches, scanned = {}, pd.DataFrame(), []
    if targets is not None and len(targets):
        matches, scanned = scan_primaries(
            targets, [(p, primary_root / (p + "_prim.csv")) for p in primary_names],
            info["primary_sha256"])
        for cand in binding_candidates(targets, matches, sorted(contents)):
            cand["primary"] = (cand["primary_members"][0]
                               if cand["state"] == "exact_unique_file_candidate" else None)
            binding[cand["detector_member"]] = cand
    for cid in contents:
        binding.setdefault(cid, {"detector_member": cid, "eligible_records": 0, "matched_records": 0,
                                 "ambiguous_records": 0, "primary_members": [],
                                 "all_row_id_equal": False, "state": "unresolved", "primary": None})
        contents[cid]["binding"] = {k: v for k, v in binding[cid].items() if k != "detector_member"}
        want = {pinned_primary[n] for n in contents[cid]["members"] if pinned_primary.get(n) is not None}
        contents[cid]["pinned_primary"] = sorted(want)[0] if want else None
        if len(want) > 1 or (want and binding[cid]["primary"] not in want):
            raise CensusStop(f"Pinned binding not reproduced for {cid}: {sorted(want)} vs "
                             f"{binding[cid]['primary']}")
    summary["timings"]["binding"] = {"seconds": time.monotonic() - t2, "primary_files": len(scanned),
                                     "bytes": sum(f["bytes"] for f in scanned),
                                     "target_records": 0 if targets is None else int(len(targets))}
    summary["primary_files"] = scanned

    # Union-find groups over every identity link.
    t3 = time.monotonic()
    uf = UnionFind(sorted(contents))
    edges = []
    for side in ("hits_sha256", "vacs_sha256"):
        first = {}
        for cid in sorted(contents):
            other = first.setdefault(contents[cid][side], cid)
            if other != cid:
                edges.append((other, cid, "bytes"))
    first = {}
    for cid in sorted(contents):
        for s in set(contents[cid]["signatures"]):
            other = first.setdefault(s, cid)
            if other != cid:
                edges.append((other, cid, "event_signature"))
    by_primary = defaultdict(set)
    for cid, p in primary_links(matches, binding):
        by_primary[p].add(cid)
    for p, group in by_primary.items():
        group = sorted(group)
        edges.extend((group[0], cid, "primary_file") for cid in group[1:])
    pairs = shared_record_pairs(records[["content", "Row", "h"]])
    summary["shared_record_histogram"] = {str(k): int(v) for k, v in
                                          pairs.value_counts().sort_index().items()}
    for (a, _, b, _), n in pairs.items():
        if n >= MIN_SHARED_RECORDS:
            edges.append((a, b, "primary_records"))
    for a, b, _ in edges:
        uf.union(a, b)
    evidence = defaultdict(Counter)
    for a, _, kind in edges:
        evidence[uf.find(a)][kind] += 1
    members_of = defaultdict(list)
    for cid in sorted(contents):
        members_of[uf.find(cid)].append(cid)

    groups = []
    for root, cids in sorted(members_of.items()):
        kept = min(cids, key=lambda c: (-contents[c]["events"], c))
        bound = {binding[c]["primary"] for c in cids if binding[c]["primary"] is not None}
        group = {"id": f"{tag}:{kept}", "kept": kept, "contents": cids,
                 "files": sum(len(contents[c]["members"]) for c in cids),
                 "events": contents[kept]["events"], "primary": binding[kept]["primary"],
                 "truth_bound": binding[kept]["primary"] is not None,
                 "evidence": dict(sorted(evidence[root].items())), "dropped": []}
        pinned_here = [c for c in cids if contents[c]["pinned_primary"] is not None]
        group["pinned_member_kept"] = (kept in pinned_here) if pinned_here else None
        if len(bound) > 1:
            summary["stop_reasons"].append({"kind": "several_primaries", "group": group["id"],
                                            "primaries": sorted(bound)})
        elif bound and not group["truth_bound"]:
            summary["stop_reasons"].append({"kind": "kept_content_unbound", "group": group["id"],
                                            "bound_primary": sorted(bound)[0]})
        for c in cids:
            if c == kept:
                continue
            relation, detail = contained(contents[c], contents[kept], raw_root)
            group["dropped"].append({"content": c, "relation": relation, **detail})
            if relation is None:
                summary["stop_reasons"].append({"kind": "dropped_not_subset", "group": group["id"],
                                                "content": c, "kept": kept, **detail})
        groups.append(group)
    summary["timings"]["groups"] = {"seconds": time.monotonic() - t3, "groups": len(groups)}

    kept_ids = {g["kept"] for g in groups}
    if targets is not None:
        summary["vertex_spread_mc_truth"] = vertex_spread(
            targets.loc[targets.detector_member.isin(kept_ids)])
    events_path, records_path = output / "events.csv.gz", output / "records.csv.gz"
    write_index(events_path, events_index)
    with gzip.open(records_path, "wt") as stream:
        stream.write("content,Row,Id,sha256\n")
        for row in records.itertuples(index=False):
            stream.write(f"{row.content},{row.Row},{row.Id},{row.h}\n")
    summary["indexes"] = {
        "events": {"path": events_path.name, "sha256": sha256(events_path), "rows": len(events_index),
                   "columns": "group(tag),file(content),Row,records(hits),sha256(event signature)"},
        "records": {"path": records_path.name, "sha256": sha256(records_path), "rows": int(len(records)),
                    "columns": "content,Row,Id,sha256(direct-primary record over mapper keys)"}}
    summary["header_only"] = header_only
    summary["files"] = {n: [f["hits"], f["vacs"], f["hits_bytes"], f["vacs_bytes"]] for n, f in files.items()}
    summary["contents"] = {cid: {k: v for k, v in c.items()
                                 if k not in ("signatures", "records", "final_signature", "id")}
                           for cid, c in sorted(contents.items())}
    summary["groups"] = groups
    summary["counts"] = {"contents": len(contents), "groups": len(groups),
                         "truth_bound_groups": sum(g["truth_bound"] for g in groups),
                         "kept_events": sum(g["events"] for g in groups),
                         "events_all_contents": sum(c["events"] for c in contents.values())}
    summary["timings"]["total_seconds"] = time.monotonic() - t0


def stat_projection(info):
    """Cheap full-population sizes for the smoke runtime projection."""
    raw_root, primary_root = Path(info["raw_root"]), Path(info["primary_root"])
    sizes, raw_bytes = {}, 0
    for n in info["all_pair_names"]:
        pair = tuple(stat_size(raw_root / (n + s)) for s in ("_hits.csv", "_vacs.csv"))
        raw_bytes += sum(pair)
        if pair != (EMPTY_BYTES["hits"], EMPTY_BYTES["vacs"]):
            sizes[pair] = sizes.get(pair, 0) + 1
    prim = sum(stat_size(primary_root / (p + "_prim.csv")) for p in info["common_primary_pair_names"])
    # Byte-unique contents are approximated by distinct size pairs (upper bound
    # on the parse count only if no size coincidences hide distinct contents).
    return {"raw_bytes": raw_bytes, "size_groups": len(sizes), "primary_bytes": prim,
            "primary_files": len(info["common_primary_pair_names"])}


def project(summary, full):
    t = summary["timings"]
    hash_rate = t["hash"]["seconds"] / max(t["hash"]["bytes"], 1)
    content_rate = t["contents"]["seconds"] / max(t["contents"]["contents"], 1)
    scan_rate = t["binding"]["seconds"] / max(t["binding"]["bytes"], 1)
    group_rate = t["groups"]["seconds"] / max(t["contents"]["contents"], 1)
    # Raw files are hashed once and the byte-unique contents once more after parsing.
    seconds = (hash_rate * full["raw_bytes"] + (content_rate + group_rate) * full["size_groups"]
               + scan_rate * full["primary_bytes"])
    return {"basis": full, "rates": {"hash_s_per_byte": hash_rate, "content_s": content_rate,
                                     "scan_s_per_byte": scan_rate, "group_s_per_content": group_rate},
            "estimated_seconds": seconds,
            "estimated_seconds_with_margin": MARGIN_FACTOR * seconds + MARGIN_SECONDS}


def smoke_primaries(info, members):
    bound = sorted({info["primary_member_map"][m] for m in members
                    if info["primary_member_map"].get(m) is not None})
    other = next(p for p in sorted(info["common_primary_pair_names"]) if p not in bound)
    return bound + [other]


def check_smoke(summary, info, members):
    """The smoke must show one alias content, one header-only and the pinned binding."""
    bound, alias, header = members
    contents = summary.get("contents", {})
    if summary["status"] != "complete":
        return
    if (summary.get("header_only") != [header] or len(contents) != 1
            or contents.get(min(bound, alias), {}).get("members") != sorted([bound, alias])
            or contents[min(bound, alias)]["binding"]["primary"] != info["primary_member_map"][bound]
            or summary["counts"]["truth_bound_groups"] != 1):
        summary["stop_reasons"].append({"kind": "smoke_expectation",
                                        "detail": "alias, header-only or pinned binding not reproduced"})
        summary["status"] = "stopped"


def combine(tag_dirs, output, *, coverage, expected_pinned_total=None):
    """Cross-potential overlap check and the compact census for committing."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    census = {"schema": SCHEMA + "-compact", "coverage": coverage, "tags": {}, "stop_reasons": [],
              "units": {"events": "exported events with at least one raw hit (distinct Row)"}}
    loaded, sig_owner, records, primaries, byte_owner = {}, {}, [], defaultdict(set), defaultdict(set)
    common, shared_events, examples = None, 0, []
    for tag in TAGS:
        if tag not in tag_dirs:
            continue
        d = Path(tag_dirs[tag])
        s = json.loads((d / "summary.json").read_text())
        if s["status"] != "complete" or s["tag"] != tag:
            raise CensusStop(f"Tag census not complete: {tag}")
        if coverage == "full" and s["coverage"] != "full":
            raise CensusStop(f"Full census combined with a subset: {tag}")
        binding = {k: s.get(k) for k in ("inventory_sha256", "code_sha256", "source_digest")}
        if common is None:
            common = binding
        elif binding != common:
            raise CensusStop("Tag censuses come from different code or inventories")
        for kind, item in s["indexes"].items():
            if sha256(d / item["path"]) != item["sha256"]:
                raise CensusStop(f"Census index changed: {tag}/{item['path']}")
            target = output / f"{tag}_{item['path']}"
            if target.resolve() != (d / item["path"]).resolve():
                shutil.copyfile(d / item["path"], target)
            if sha256(target) != item["sha256"]:
                raise CensusStop("Index copy differs")
        loaded[tag] = s
        with gzip.open(d / s["indexes"]["events"]["path"], "rt") as stream:
            next(stream)
            for line in stream:
                row = line.rstrip("\n").split(",")
                owner = sig_owner.setdefault(row[-1], tag)
                if owner != tag:
                    shared_events += 1
                    if len(examples) < 5:
                        examples.append({"signature": row[-1], "first_tag": owner,
                                         "tag": tag, "content": row[1], "Row": row[2]})
        r = pd.read_csv(d / s["indexes"]["records"]["path"], dtype={"content": str})
        r["content"] = tag + ":" + r["content"]
        records.append(r.rename(columns={"sha256": "h"})[["content", "Row", "h"]])
        for f in s["primary_files"]:
            primaries[f["sha256"]].add(tag)
        for c in s["contents"].values():
            byte_owner[c["hits_sha256"]].add(tag)
            byte_owner[c["vacs_sha256"]].add(tag)
    census.update(common or {})
    pairs = shared_record_pairs(pd.concat(records, ignore_index=True)) if records else pd.Series(dtype=int)
    if len(pairs):
        tags_of = pairs.index.to_frame(index=False)
        cross = pairs[(tags_of.a.str.split(":").str[0] != tags_of.b.str.split(":").str[0]).to_numpy()]
    else:
        cross = pairs
    census["cross_potential"] = {
        "shared_event_signatures": shared_events, "shared_event_examples": examples,
        "shared_record_histogram": {str(k): int(v) for k, v in cross.value_counts().sort_index().items()},
        "event_pairs_with_min_shared_records": int((cross >= MIN_SHARED_RECORDS).sum()),
        "shared_raw_files": sum(len(v) > 1 for v in byte_owner.values()),
        "shared_primary_files": sum(len(v) > 1 for v in primaries.values())}
    x = census["cross_potential"]
    if shared_events:
        census["stop_reasons"].append({"kind": "cross_potential_events", "events": shared_events})
    if x["event_pairs_with_min_shared_records"]:
        census["stop_reasons"].append({"kind": "cross_potential_records",
                                       "pairs": x["event_pairs_with_min_shared_records"]})
    if x["shared_raw_files"] or x["shared_primary_files"]:
        census["stop_reasons"].append({"kind": "cross_potential_bytes",
                                       "raw": x["shared_raw_files"], "primary": x["shared_primary_files"]})
    if expected_pinned_total is not None and sum(
            s["pinned"]["bound_members_verified"] for s in loaded.values()) != expected_pinned_total:
        census["stop_reasons"].append({"kind": "pinned_bound_total"})
    for tag, s in loaded.items():
        census["tags"][tag] = {
            "raw_root": s["raw_root"], "primary_root": s["primary_root"], "coverage": s["coverage"],
            "membership": s["membership"], "pinned": s["pinned"], "counts": s["counts"],
            "header_only": s["header_only"],
            "contents": {cid: {k: c[k] for k in ("members", "hits_sha256", "vacs_sha256", "events",
                                                  "hit_records", "direct_primary_records",
                                                  "signature_digest", "pinned_primary")}
                         | {"binding": {k: c["binding"][k] for k in ("state", "primary", "eligible_records",
                                                                      "matched_records", "ambiguous_records")}}
                         for cid, c in s["contents"].items()},
            "groups": s["groups"], "shared_record_histogram": s["shared_record_histogram"],
            "vertex_spread_mc_truth": s.get("vertex_spread_mc_truth"),
            "primary_files": {f["member"]: f["sha256"] for f in s["primary_files"]},
            "indexes": {k: {"path": f"{tag}_{v['path']}", "sha256": v["sha256"], "rows": v["rows"]}
                        for k, v in s["indexes"].items()},
            "tag_summary_sha256": sha256(Path(tag_dirs[tag]) / "summary.json")}
    census["status"] = "stopped" if census["stop_reasons"] else "complete"
    write_json(output / "census.json", census)
    return census


def validate_smoke_receipt(path, source, environment):
    receipt = json.loads(with_access_retry(Path.read_text, Path(path)))
    if (receipt.get("status") != "verified" or receipt.get("mode") != "smoke"
            or receipt.get("slurm_state") != "COMPLETED" or receipt.get("exit_code") != "0:0"
            or receipt.get("source_digest") != source or receipt.get("environment") != environment):
        raise ValueError(f"Invalid smoke receipt: {path}")
    if read_sha256(receipt["archive"]) != receipt["archive_sha256"]:
        raise ValueError("Verified smoke archive changed")
    if max(receipt["estimated_full_seconds_with_margin"].values()) > FULL_LIMIT_SECONDS:
        raise ValueError("Measured smoke does not support the 1:30:00 limit")
    return receipt


def run(args):
    started = time.monotonic()
    source = source_digest(args.source)
    environment = runtime()
    inventory_path = args.source / INVENTORY
    inventory = json.loads(inventory_path.read_text())
    if args.mode == "full":
        validate_smoke_receipt(args.smoke_receipt, source, environment)
    args.output.mkdir(parents=True, exist_ok=False)
    meta = {"inventory_sha256": sha256(inventory_path), "code_sha256": sha256(Path(__file__)),
            "source_digest": source}
    tags = TAGS if args.mode == "smoke" else (TAGS[args.task],)
    summary = {"schema": SCHEMA + "-job", "mode": args.mode, "job_id": args.job_id, "tags": {},
               "environment": environment, **meta}
    for tag in tags:
        info = inventory["tags"][tag]
        members = SMOKE_MEMBERS[tag] if args.mode == "smoke" else None
        result = census_tag(info, tag, args.output / tag, members=members,
                            primaries=smoke_primaries(info, members) if members else None,
                            workers=args.workers, meta=meta,
                            expected_pinned_bound=None if members else PINNED_BOUND[tag])
        entry = {"status": result["status"], "stop_reasons": result["stop_reasons"][:20],
                 "counts": result.get("counts"), "timings": result["timings"],
                 "access_retries": len(result["access_retries"])}
        if args.mode == "smoke":
            check_smoke(result, info, members)
            write_json(args.output / tag / "summary.json", result)
            entry.update(status=result["status"], stop_reasons=result["stop_reasons"][:20])
            if result["status"] == "complete":
                entry["projection"] = project(result, stat_projection(info))
        summary["tags"][tag] = entry
        print(json.dumps({"tag": tag, "status": result["status"],
                          "seconds": result["timings"].get("total_seconds")}), flush=True)
    complete = all(v["status"] == "complete" for v in summary["tags"].values())
    if args.mode == "smoke" and complete:
        census = combine({t: args.output / t for t in tags}, args.output, coverage="smoke")
        summary["census"] = {"status": census["status"], "sha256": sha256(args.output / "census.json"),
                             "cross_potential": census["cross_potential"]}
        complete = census["status"] == "complete"
    if source_digest(args.source) != source:
        raise ValueError("Census source changed during execution")
    summary["status"] = "complete" if complete else "stopped"
    summary["seconds"] = time.monotonic() - started
    write_json(args.output / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("mode", "status", "seconds", "job_id")}), flush=True)
    return 0 if complete else STOPPED


def verify_export(args):
    """Local: hash before extraction, then bind the export to Slurm and source."""
    slurm = json.loads(args.slurm.read_text())
    archive_sha = sha256(args.archive)
    if archive_sha != args.expected_sha256:
        raise ValueError("Downloaded archive differs from the hash printed by the job")
    if slurm["state"] != "COMPLETED" or slurm["exit_code"] != "0:0":
        raise ValueError("Only terminal successful Slurm jobs are verified")
    source = source_digest(args.source)
    args.extracted.mkdir(parents=True, exist_ok=False)
    with tarfile.open(args.archive) as tar:
        tar.extractall(args.extracted, filter="data")
    result = args.extracted / "result"
    summary = json.loads((result / "summary.json").read_text())
    if (summary["job_id"] != slurm["job_key"] or summary["status"] != "complete"
            or summary["source_digest"] != source or summary["mode"] != args.mode):
        raise ValueError("Export does not bind the declared Slurm job, source or mode")
    for tag in summary["tags"]:
        s = json.loads((result / tag / "summary.json").read_text())
        if s["status"] != "complete" or s["source_digest"] != source:
            raise ValueError("Tag census incomplete: " + tag)
        for item in s["indexes"].values():
            if sha256(result / tag / item["path"]) != item["sha256"]:
                raise ValueError("Exported index corrupt: " + tag)
    receipt = {"status": "verified", "mode": args.mode, "job_id": summary["job_id"],
               "slurm_state": slurm["state"], "exit_code": slurm["exit_code"],
               "elapsed": slurm.get("elapsed"), "node": slurm.get("node"),
               "source_digest": source, "environment": summary["environment"],
               "archive": args.cluster_archive, "archive_sha256": archive_sha,
               "local_archive": str(args.archive), "summary_sha256": sha256(result / "summary.json")}
    if args.mode == "smoke":
        if summary["census"]["status"] != "complete":
            raise ValueError("Smoke combine did not complete")
        receipt["census_sha256"] = sha256(result / "census.json")
        receipt["estimated_full_seconds"] = {t: v["projection"]["estimated_seconds"]
                                             for t, v in summary["tags"].items()}
        receipt["estimated_full_seconds_with_margin"] = {
            t: v["projection"]["estimated_seconds_with_margin"] for t, v in summary["tags"].items()}
        if max(receipt["estimated_full_seconds_with_margin"].values()) > FULL_LIMIT_SECONDS:
            raise ValueError("Measured smoke does not support the 1:30:00 limit")
    write_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="cluster job: smoke (all tags, subset) or full (one tag)")
    r.add_argument("--source", type=Path, required=True)
    r.add_argument("--mode", choices=["smoke", "full"], required=True)
    r.add_argument("--task", type=int, choices=range(len(TAGS)), default=0)
    r.add_argument("--smoke-receipt", type=Path)
    r.add_argument("--job-id", required=True)
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    v = sub.add_parser("verify-export", help="local: verify a downloaded export")
    v.add_argument("--archive", type=Path, required=True)
    v.add_argument("--expected-sha256", required=True)
    v.add_argument("--slurm", type=Path, required=True, help="json: job_key, state, exit_code, elapsed, node")
    v.add_argument("--source", type=Path, required=True, help="local copy of the immutable package")
    v.add_argument("--mode", choices=["smoke", "full"], required=True)
    v.add_argument("--extracted", type=Path, required=True)
    v.add_argument("--receipt", type=Path, required=True)
    v.add_argument("--cluster-archive", required=True)
    c = sub.add_parser("combine", help="local: cross-potential check and compact census")
    c.add_argument("--tag-result", action="append", required=True, help="tag=extracted result/<tag> dir")
    c.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    if args.command == "run":
        if args.mode == "full" and args.smoke_receipt is None:
            p.error("--smoke-receipt is required for the full census")
        return run(args)
    if args.command == "verify-export":
        return verify_export(args)
    dirs = dict(x.split("=", 1) for x in args.tag_result)
    if set(dirs) != set(TAGS):
        p.error("all three tag results are required")
    census = combine(dirs, args.output, coverage="full", expected_pinned_total=sum(PINNED_BOUND.values()))
    print(json.dumps({"status": census["status"], "cross_potential": census["cross_potential"]}))
    return 0 if census["status"] == "complete" else STOPPED


if __name__ == "__main__":
    sys.exit(main())
