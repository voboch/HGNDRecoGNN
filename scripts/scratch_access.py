"""Bounded retry for /scratch access that fails with EACCES (RA-09).

On cHARISMa, opening or stat-ing a readable file under /scratch/vbocharnikov
has failed with EACCES at a different file and node each time; 18 events
between 2026-10-17 and 2026-10-28, two of them on login nodes. In the census
4464531 every one of four such failures cleared on the first retry after 1 s,
and two struck different nodes within one second. The cause is not
established (protocol amendment 2026-10-28, reports/smash_train_protocol_20261028.md).

The waits are a convention, not a measurement: the census recorded that a
1 s wait sufficed four times out of four, nothing about longer outages. The
worst case adds 63 s per access; a longer outage still raises, as before.
Only PermissionError with errno EACCES is retried. Every retry is printed to
stderr and appended to ACCESS_RETRIES so a job can write it into its receipt.
A retried read cannot change a result unnoticed where the caller hashes what
it read, which every caller in the SMASH training path does.
"""

from __future__ import annotations

import errno
import json
import os
import sys
import time

ACCESS_RETRY_WAITS = (1, 2, 4, 8, 16, 32)
ACCESS_RETRIES = []


def with_access_retry(call, *args, cleanup=None, **kwargs):
    """call(*args, **kwargs), retried on EACCES only.

    `cleanup` runs before each retry and must undo what a failed attempt may
    have written (a partial copy opened with mode 'x', say); an EACCES inside
    it counts as one more failed attempt.
    """
    for attempt, wait in enumerate((*ACCESS_RETRY_WAITS, None), start=1):
        try:
            if attempt > 1 and cleanup is not None:
                cleanup()
            return call(*args, **kwargs)
        except PermissionError as error:
            if wait is None or error.errno != errno.EACCES:
                raise
            event = {"path": error.filename and str(error.filename),
                     "call": getattr(call, "__qualname__", repr(call)), "attempt": attempt,
                     "wait_seconds": wait, "node": os.uname().nodename,
                     "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            ACCESS_RETRIES.append(event)
            print("access retry " + json.dumps(event), file=sys.stderr, flush=True)
            time.sleep(wait)
