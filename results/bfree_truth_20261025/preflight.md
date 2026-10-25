# B-free CPU preflight, 2026-10-25

Independent reviewer: /root/v9_bfree_preflight; verdict GO-with-fixes.
Scientific path accepted: fixed inherited selections, B ignored; exact primary
counter replay, full-file resampling shared between spectra and counters.
The reviewer verified all568members/2345993events against the census and all
nine declared smoke members. Focused tests13passed; full suite225passed.

Mechanical conditions before smoke: commit plan/code, checksum-pin complete
source package, record exact Slurm commands. Full submission additionally
requires externally verified successful smoke artifacts and measured runtime
estimate with margin <=45minutes per potential. Full command overrides:
--partition=rocky --time=00:45:00 --array=0-2%2. No automatic retry.
