#ifndef BMN_TRACK_EXPORT_H
#define BMN_TRACK_EXPORT_H
//
// Reconstructed-track and event-ledger export for the BM@N DST converter.
//
// Included by ncx_refactor/Convert2Csv.C and ncx/convert2csv.cc so both run one
// DST event loop and emit the HGND CSVs and the charged-arm tables together.
// Nothing here duplicates the converter: it owns the track/event schemas and
// the DST getters, and the converter owns the loop.
//
// ---------------------------------------------------------------------------
// VERIFIED API -- checked against the installed BmnRoot source on ncx
// ---------------------------------------------------------------------------
// Read off /lhep/users/boch/scratch/bmnroot (= /scratch2/vboch/bmnroot, the
// VMCWORKDIR in config.sh) on 2026-10-13: base/data/BmnGlobalTrack.{h,cxx},
// base/data/BmnTrack.h, base/cbm/CbmStsTrack.h and
// reconstruction/globaltracking/BmnGlobalTracking.cxx. Still NOT executed:
// run_convert.sh defaults BMNROOT_CONFIG to a DIFFERENT build
// (/nica/bmn1/karpushkin/.../bmnroot_dev), so inspect_tracks.C re-checks every
// gate below at run time and exits non-zero rather than writing a column it
// cannot fill honestly.
//
//   branches        "BmnGlobalTrack" (GLOBAL), "StsTrack" (CbmStsTrack, the
//                   inner-track array -- BmnGlobalTracking reads it through
//                   fInnerTrackBranchName), "BmnTof400Hit", "BmnTof700Hit"
//   charge sign     GetParamFirst()->GetQp()   signed q/p; sign is the charge
//                   BmnTrack::GetP() is 1.0/GetQp() and is therefore SIGNED;
//                   never use it as a momentum magnitude
//   momentum        p = 1/|q/p|
//   charge MAGNITUDE
//                   there is no GetQ(). GetZ() exists but BmnGlobalTracking
//                   ::CalcdQdn assigns it from a hardcoded He/Z=1 dE/dx cut
//                   (20000*exp(-2*sqrt(p))+600), so it is a DERIVED flag from
//                   the same dE/dx, not an independent charge measurement.
//                   Exported as z_flag for QA; never an input to selection.
//   dE/dx           GetdQdNLower(). There is no GetDedx() on BmnGlobalTrack or
//                   BmnTrack. Units are calibrated GEM cluster charge (ADC-
//                   like, ~1e2-1e4), NOT MeV/cm, so the Bethe parametrisation
//                   must be fitted in these units and cannot be imported.
//                   Two distinct invalid values, both exported as invalid:
//                     0.0   ctor default -- CalcdQdn never ran for this track
//                     -100  ran but <3 usable GEM hits (usedHits == 0)
//   dE/dx ASYMMETRY CalcdQdn returns early when GetP() < 0, i.e. NEGATIVE
//                   tracks never receive dE/dx at all. A pi- reference band is
//                   therefore unavailable from this branch; the r_pi > 3 veto
//                   can only be built from pi+ on the positive side. This is a
//                   physics-relevant limit on the PID, measured by the probe.
//   TOF match       GetTof1HitIndex() / GetTof2HitIndex()
//                   no-argument form = COMMON matching, not a proton hypothesis
//   m^2             GetMass2(1) / GetMass2(2); returns kM2Invalid when the
//                   corresponding beta is unset (fBeta400/700 default -1000).
//                   NOTE both are NON-const members, as are GetZ/GetPx/...,
//                   so the track pointer here cannot be const.
//   primary flag    BmnGlobalTrack::IsPrimary() (fIsPrimary). CbmStsTrack has
//                   no FromVertex(). CAUTION: the constructor sets
//                   fIsPrimary(kTRUE), so the flag reads "primary" until
//                   something sets it false. On the SMASH defaultSpot DSTs it
//                   is true for 100 % of tracks, i.e. it carries NO
//                   information there and must not be used as a primary cut.
//                   from_vertex is exported as recorded; interpreting it is
//                   the caller's job, and a 100 % rate means "not set".
//   inner branch    "StsTrack" OR "StsVector" -- fInnerTrackBranchName
//                   defaults to the former but BmnStsVectorFinder registers
//                   the latter, and both are TClonesArray("CbmStsTrack"). The
//                   converter tries both.
//   inner link      GetGemTrackIndex() is set only inside BmnGlobalTracking's
//                   `if (fStsTracks)` block, which also holds CalcdQdn and the
//                   vertex DCA/chi2 update. When the inner branch name did not
//                   match at reconstruction time that block was skipped, and
//                   the link is -1 on every track -- so inner_hits, mc_index,
//                   dedx, dca_cm and vertex chi2 are ALL unavailable together.
//                   That is one root cause with five symptoms, not five
//                   separate defects.
//   inner hits      CbmStsTrack::GetNStsHits() (not GetNHits())
//   MC assoc        CbmStsTrack::GetTrkID(); the global GetRefIndex() is
//                   documented as always -1
//
// Any of these being absent or renamed is a reason to stop and inspect, not to
// substitute. In particular nothing here falls back to MC truth: `mc_*` columns
// are written for validation only and carry a `mc_` prefix so no selection can
// read them by accident.
//
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

// tests/cpp/test_track_export.cc defines BMNX_STANDALONE_TEST and supplies
// minimal stubs for these, so the extraction arithmetic can be compiled and
// exercised without a ROOT installation. Production builds take the real ones.
#ifndef BMNX_STANDALONE_TEST
#include "TString.h"
#include "TClonesArray.h"
#include "TSystem.h"
#endif

namespace bmnx {

// Sentinel the DST uses for an unmatched/invalid mass-squared.
constexpr double kM2Invalid = -1000.0;
// What we write when a quantity is absent. Distinct from any physical value and
// from the DST's own sentinel, so "missing" is never confused with "measured".
constexpr double kNA = -9999.0;

// ---------------------------------------------------------------------------
// CSV writer. Shared so the track tables format exactly like the HGND ones.
// ---------------------------------------------------------------------------
class CsvWriter {
 public:
  CsvWriter() = default;

  bool open(const TString& path, const std::string& header) {
    path_ = path;
    out_.open(path.Data());
    if (!out_.is_open()) return false;
    out_ << header << '\n';
    out_.precision(9);
    return out_.good();
  }

  bool ok() const { return out_.is_open() && out_.good(); }
  const TString& path() const { return path_; }
  Long64_t rows() const { return rows_; }
  void close() { if (out_.is_open()) out_.close(); }

  void row(const std::vector<long>& ints, const std::vector<double>& nums) {
    for (size_t i = 0; i < ints.size(); ++i) {
      if (i) out_ << ',';
      out_ << ints[i];
    }
    for (size_t i = 0; i < nums.size(); ++i) out_ << ',' << nums[i];
    out_ << '\n';
    ++rows_;
  }

 private:
  TString       path_;
  std::ofstream out_;
  Long64_t      rows_ = 0;
};

// ---------------------------------------------------------------------------
// Schemas
// ---------------------------------------------------------------------------
// Row is the ZERO-BASED TTree entry, identical to the hits/vacs/prim CSVs, so
// (source_id, Row) joins the charged arm to the HGND arm. source_id is a stable
// hash of the input path: Row alone is unique only within a file.
//
// Both TOF arms are carried. A calibration or selection uses ONE explicitly,
// because a union would count a track matched in both arms twice.
inline const char* kTracksHeader() {
  return "source_id,Row,Instance,track_index,inner_index,"
         "charge_sign,q_over_p,p,px,py,pz,pt,eta,phi,"
         "dedx,dedx_valid,dedx_status,z_flag,"
         "from_vertex,inner_hits,chi2_ndf,dca_cm,"
         "tof400_index,tof400_valid,m2_400,beta_400,path_400,time_400,"
         "tof700_index,tof700_valid,m2_700,beta_700,path_700,time_700,"
         "mc_index,mc_pdg,mc_mother";
}

// One row per processed entry, written even when the event has no tracks and
// no selected protons. Without it an efficiency denominator silently loses the
// events that failed reconstruction, which are exactly the ones that matter.
inline const char* kEventsHeader() {
  return "source_id,Row,event_id,n_tracks,n_tracks_vertex,"
         "n_tof400,n_tof700,n_dedx_valid,vertex_valid,vertex_chi2,"
         "vx,vy,vz,B,NPrim";
}

// ---------------------------------------------------------------------------
// Stable source identifier. Row repeats across files by design; the pair
// (source_id, Row) does not. FNV-1a over the basename, so moving the tree does
// not renumber events, and it is reproducible in Python for the join tests.
// ---------------------------------------------------------------------------
inline long SourceId(const TString& path) {
  TString base = gSystem ? TString(gSystem->BaseName(path.Data())) : path;
  unsigned long long h = 1469598103934665603ULL;
  for (int i = 0; i < base.Length(); ++i) {
    h ^= static_cast<unsigned char>(base[i]);
    h *= 1099511628211ULL;
  }
  // keep it positive and comfortably inside a CSV integer / int64 column
  return static_cast<long>(h % 2147483647ULL);
}

// ---------------------------------------------------------------------------
// What one reconstructed track contributes. Everything defaults to "missing",
// so a getter that cannot be filled stays missing rather than becoming zero.
// ---------------------------------------------------------------------------
struct TrackRow {
  long   track_index   = -1;
  long   inner_index   = -1;
  long   charge_sign   = 0;        // +1 / -1 from the sign of q/p; 0 = unknown
  double q_over_p      = kNA;
  double p = kNA, px = kNA, py = kNA, pz = kNA, pt = kNA, eta = kNA, phi = kNA;
  double dedx          = kNA;
  long   dedx_valid    = 0;
  double dedx_status   = kNA;      // raw GetdQdNLower() before validity gating
  double z_flag       = kNA;      // GetZ(): derived He flag, NOT a charge magnitude
  long   from_vertex   = -1;       // -1 unknown, 0 no, 1 yes
  long   inner_hits    = -1;
  double chi2_ndf      = kNA;
  double dca_cm        = kNA;
  long   tof_index[2]  = {-1, -1};
  long   tof_valid[2]  = {0, 0};
  double m2[2]         = {kNA, kNA};
  double beta[2]       = {kNA, kNA};
  double path[2]       = {kNA, kNA};
  double time[2]       = {kNA, kNA};
  // validation only -- never read by a selection
  long   mc_index      = -1;
  long   mc_pdg        = 0;
  long   mc_mother     = -2;       // -1 is "primary"; -2 is "not associated"

  void write(CsvWriter& w, long source_id, Long64_t row, long instance) const {
    w.row({source_id, static_cast<long>(row), instance, track_index, inner_index,
           charge_sign},
          {q_over_p, p, px, py, pz, pt, eta, phi,
           dedx, static_cast<double>(dedx_valid), dedx_status, z_flag,
           static_cast<double>(from_vertex), static_cast<double>(inner_hits),
           chi2_ndf, dca_cm,
           static_cast<double>(tof_index[0]), static_cast<double>(tof_valid[0]),
           m2[0], beta[0], path[0], time[0],
           static_cast<double>(tof_index[1]), static_cast<double>(tof_valid[1]),
           m2[1], beta[1], path[1], time[1],
           static_cast<double>(mc_index), static_cast<double>(mc_pdg),
           static_cast<double>(mc_mother)});
  }
};

struct EventRow {
  long   event_id        = -1;
  long   n_tracks        = 0;
  long   n_tracks_vertex = 0;
  long   n_tof[2]        = {0, 0};
  long   n_dedx_valid    = 0;
  long   vertex_valid    = 0;
  double vertex_chi2     = kNA;
  double vx = kNA, vy = kNA, vz = kNA;
  double B               = -1.0;
  long   NPrim           = -1;

  void write(CsvWriter& w, long source_id, Long64_t row) const {
    w.row({source_id, static_cast<long>(row), event_id, n_tracks,
           n_tracks_vertex, n_tof[0], n_tof[1], n_dedx_valid, vertex_valid},
          {vertex_chi2, vx, vy, vz, B, static_cast<double>(NPrim)});
  }
};

// ---------------------------------------------------------------------------
// Fill one TrackRow from a BmnGlobalTrack. Templated on the track type so this
// header does not force the converter to include BmnRoot headers it may not
// need; the converter instantiates it with the real class.
//
// `inner_tracks` may be null -- then vertex association and the MC link are
// left unknown rather than guessed.
// ---------------------------------------------------------------------------
template <class GlobalTrack, class InnerTrack>
TrackRow FillTrackRow(GlobalTrack* trk, long index,
                      TClonesArray* inner_tracks) {
  TrackRow r;
  if (!trk) return r;
  r.track_index = index;

  // --- momentum and CHARGE SIGN from signed q/p ---------------------------
  auto* par = trk->GetParamFirst();
  if (par) {
    const double qp = par->GetQp();
    r.q_over_p = qp;
    if (qp != 0.0 && std::isfinite(qp)) {
      r.charge_sign = (qp > 0.0) ? 1 : -1;
      r.p = 1.0 / std::fabs(qp);
      const double tx = par->GetTx(), ty = par->GetTy();
      const double norm = std::sqrt(1.0 + tx * tx + ty * ty);
      r.pz = r.p / norm;
      r.px = r.pz * tx;
      r.py = r.pz * ty;
      r.pt = std::hypot(r.px, r.py);
      if (r.pt > 0.0) {
        r.phi = std::atan2(r.py, r.px);
        const double theta = std::atan2(r.pt, r.pz);
        if (theta > 0.0 && theta < M_PI)
          r.eta = -std::log(std::tan(0.5 * theta));
      }
    }
  }

  // --- energy loss, from GetdQdNLower(). Calibrated GEM cluster charge, not
  // MeV/cm. 0.0 means CalcdQdn never ran for this track (ctor default) and
  // -100 means it ran with too few usable GEM hits; both are invalid, and the
  // raw value is kept in dedx_status so the two can be told apart downstream.
  const double de = trk->GetdQdNLower();
  r.dedx_status = de;
  if (std::isfinite(de) && de > 0.0) { r.dedx = de; r.dedx_valid = 1; }
  r.z_flag = static_cast<double>(trk->GetZ());   // QA only, derived from de

  r.chi2_ndf = (trk->GetNDF() > 0)
                   ? trk->GetChi2() / static_cast<double>(trk->GetNDF()) : kNA;

  // --- primary association via the INNER track, not the global one --------
  // The primary decision lives on the GLOBAL track (set by the vertex finder);
  // CbmStsTrack has no FromVertex(). The inner track supplies the hit count and
  // the MC link only.
  r.from_vertex = trk->IsPrimary() ? 1 : 0;
  r.inner_index = trk->GetGemTrackIndex();
  if (inner_tracks && r.inner_index >= 0 &&
      r.inner_index < inner_tracks->GetEntriesFast()) {
    auto* inner = static_cast<InnerTrack*>(inner_tracks->At(r.inner_index));
    if (inner) {
      r.inner_hits = inner->GetNStsHits();
      r.mc_index   = inner->GetTrkID();   // validation only
    }
  }

  // --- TOF, both arms, common matching hypothesis -------------------------
  const long idx[2] = {static_cast<long>(trk->GetTof1HitIndex()),
                       static_cast<long>(trk->GetTof2HitIndex())};
  for (int a = 0; a < 2; ++a) {
    r.tof_index[a] = idx[a];
    if (idx[a] < 0) continue;
    const double m2 = trk->GetMass2(a + 1);
    if (std::isfinite(m2) && m2 > kM2Invalid + 1.0) {
      r.m2[a] = m2;
      r.tof_valid[a] = 1;
    }
  }
  return r;
}


// ---------------------------------------------------------------------------
// Resolve the MC link into actual truth quantities. Kept separate from
// FillTrackRow so the reconstruction path never touches the MC array: a caller
// that does not pass mc_tracks gets a row with mc_pdg = 0 and mc_mother = -2,
// which is "not associated" and cannot be mistaken for a particle.
//
// VALIDATION ONLY. mc_index alone cannot answer "is this a truth proton", so
// without this the efficiency and purity denominators are unmeasurable -- but
// nothing in the applied selection may read these columns, which is why they
// all carry the mc_ prefix and why the schema test asserts no unprefixed truth
// column exists.
//
// mc_mother semantics, which are easy to get wrong:
//   -2  not associated (no link, or the link is out of range)
//   -1  PRIMARY: CbmMCTrack::GetMotherId() returns -1 for a primary particle
//    0+ the index of the mother track -- 0 is a valid INDEX, not a primary
// ---------------------------------------------------------------------------
template <class McTrack>
void FillMcTruth(TrackRow& r, TClonesArray* mc_tracks) {
  if (!mc_tracks) return;                 // leave "not associated"
  if (r.mc_index < 0) return;
  if (r.mc_index >= mc_tracks->GetEntriesFast()) {
    // An out-of-range link is a bug somewhere upstream, not a particle. Record
    // it as unassociated rather than reading past the array.
    r.mc_index = -1;
    return;
  }
  auto* tk = static_cast<McTrack*>(mc_tracks->At(r.mc_index));
  if (!tk) { r.mc_index = -1; return; }
  r.mc_pdg    = tk->GetPdgCode();
  r.mc_mother = tk->GetMotherId();
}

}  // namespace bmnx

#endif  // BMN_TRACK_EXPORT_H
