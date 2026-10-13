// Standalone test of ncx_refactor/BmnTrackExport.h.
//
// BmnRoot is not installed locally, so the header is compiled against minimal
// stubs that mimic the getters it uses. This cannot prove the real DST classes
// behave this way -- that is what the ncx smoke is for -- but it does prove the
// extraction arithmetic, the missing-data handling and the MC independence,
// which are the parts that would otherwise only be checked by eye.
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ---- minimal ROOT / BmnRoot stubs ----------------------------------------
struct TString {
  std::string s;
  TString() = default;
  TString(const char* c) : s(c ? c : "") {}
  TString(const std::string& v) : s(v) {}
  const char* Data() const { return s.c_str(); }
  int Length() const { return (int)s.size(); }
  char operator[](int i) const { return s[i]; }
};
struct TClonesArray {
  std::vector<void*> v;
  int GetEntriesFast() const { return (int)v.size(); }
  void* At(int i) const { return (i >= 0 && i < (int)v.size()) ? v[i] : nullptr; }
};
struct TSystemStub { const char* BaseName(const char* p) { const char* s = strrchr(p,'/'); return s ? s+1 : p; } };
static TSystemStub gSystemObj; static TSystemStub* gSystem = &gSystemObj;
typedef long long Long64_t;

#define BMNX_STANDALONE_TEST 1
#include "../../ncx_refactor/BmnTrackExport.h"

// ---- fake DST objects ------------------------------------------------------
struct FakeParam {
  double qp, tx, ty;
  double GetQp() const { return qp; }
  double GetTx() const { return tx; }
  double GetTy() const { return ty; }
};
// Mirrors CbmStsTrack: GetNStsHits (not GetNHits), no FromVertex, and
// GetTrkID is a NON-const member in the real class.
struct FakeInner {
  int nhits; int trkid;
  int GetNStsHits() const { return nhits; }
  int GetTrkID() { return trkid; }
};
// Mirrors BmnGlobalTrack. GetMass2 and GetZ are deliberately NON-const, as
// they are in base/data/BmnGlobalTrack.h: a const track pointer in the
// exporter would not compile against the real build, and this stub is what
// catches that locally.
struct FakeGlobal {
  FakeParam par; bool has_par = true;
  double dedx = 0.0, chi2 = 0.0; int ndf = 0, z = 1;
  bool primary = false;
  int inner = -1, tof1 = -1, tof2 = -1;
  double m2a = bmnx::kM2Invalid, m2b = bmnx::kM2Invalid;
  FakeParam* GetParamFirst() { return has_par ? &par : nullptr; }
  double GetdQdNLower() const { return dedx; }
  int    GetZ() { return z; }
  double GetChi2() const { return chi2; }
  int    GetNDF()  const { return ndf; }
  bool   IsPrimary() const { return primary; }
  int    GetGemTrackIndex() const { return inner; }
  int    GetTof1HitIndex()  const { return tof1; }
  int    GetTof2HitIndex()  const { return tof2; }
  double GetMass2(int id) { return id == 1 ? m2a : m2b; }
};

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("  FAIL %s\n", msg); ++fails; } } while (0)
static bool close(double a, double b, double tol = 1e-9) { return std::fabs(a-b) < tol; }

int main() {
  using namespace bmnx;

  // --- charge sign comes from the SIGN of q/p, never from GetZ() ----------
  {
    FakeGlobal pos; pos.par = {+0.5, 0.0, 0.0}; pos.z = 1;
    FakeGlobal neg; neg.par = {-0.5, 0.0, 0.0}; neg.z = 1;  // same Z flag
    auto rp = FillTrackRow<FakeGlobal, FakeInner>(&pos, 0, nullptr);
    auto rn = FillTrackRow<FakeGlobal, FakeInner>(&neg, 1, nullptr);
    CHECK(rp.charge_sign == +1, "q/p > 0 must give charge_sign +1");
    CHECK(rn.charge_sign == -1, "q/p < 0 must give charge_sign -1");
    CHECK(close(rp.p, 2.0) && close(rn.p, 2.0), "p = 1/|q/p|");
    CHECK(close(rp.z_flag, rn.z_flag),
          "GetZ() is a derived dE/dx flag and cannot separate the two");
  }

  // --- momentum components from the track slopes --------------------------
  {
    FakeGlobal t; t.par = {1.0 / 3.0, 0.3, 0.4};   // p = 3
    auto r = FillTrackRow<FakeGlobal, FakeInner>(&t, 0, nullptr);
    CHECK(close(r.p, 3.0), "p = 1/|q/p|");
    const double n = std::sqrt(1 + 0.09 + 0.16);
    CHECK(close(r.pz, 3.0 / n), "pz = p / sqrt(1+tx^2+ty^2)");
    CHECK(close(r.px, r.pz * 0.3) && close(r.py, r.pz * 0.4), "px,py = pz*t");
    CHECK(close(r.pt, std::hypot(r.px, r.py)), "pt = hypot(px,py)");
    CHECK(close(r.p, std::sqrt(r.px*r.px + r.py*r.py + r.pz*r.pz), 1e-9),
          "|p| must reconstruct from its components");
  }

  // --- missing dE/dx is missing, not zero ---------------------------------
  {
    FakeGlobal t; t.par = {0.5, 0, 0}; t.dedx = 0.0;   // CalcdQdn never ran
    auto r = FillTrackRow<FakeGlobal, FakeInner>(&t, 0, nullptr);
    CHECK(r.dedx_valid == 0, "dedx=0 must not count as measured");
    CHECK(close(r.dedx, kNA), "unfilled dedx must be the missing sentinel");
    CHECK(close(r.dedx_status, 0.0), "the raw 0.0 is preserved for diagnosis");

    // CalcdQdn writes -100 when it ran but found fewer than 3 usable GEM hits.
    // That is a different failure from "never ran" and must stay separable.
    t.dedx = -100.0;
    r = FillTrackRow<FakeGlobal, FakeInner>(&t, 0, nullptr);
    CHECK(r.dedx_valid == 0, "the -100 too-few-GEM-hits sentinel is not a value");
    CHECK(close(r.dedx_status, -100.0), "-100 must be distinguishable from 0");

    t.dedx = 2500.0;                   // GEM cluster charge, ADC-like units
    r = FillTrackRow<FakeGlobal, FakeInner>(&t, 0, nullptr);
    CHECK(r.dedx_valid == 1 && close(r.dedx, 2500.0), "real dedx is kept");

    // BmnGlobalTracking::CalcdQdn returns early for GetP() < 0, so negative
    // tracks arrive with the ctor default. The exporter must report that as
    // missing rather than inventing a value for the pi- reference band.
    FakeGlobal neg; neg.par = {-0.5, 0, 0}; neg.dedx = 0.0;
    auto rn = FillTrackRow<FakeGlobal, FakeInner>(&neg, 0, nullptr);
    CHECK(rn.charge_sign == -1 && rn.dedx_valid == 0,
          "negative tracks carry no dE/dx and must not be given one");
  }

  // --- TOF: unmatched arm, and the DST's own -1000 sentinel ---------------
  {
    FakeGlobal t; t.par = {0.5, 0, 0};
    t.tof1 = -1;                       // no match in 400
    t.tof2 = 7; t.m2b = kM2Invalid;    // matched index but invalid m2
    auto r = FillTrackRow<FakeGlobal, FakeInner>(&t, 0, nullptr);
    CHECK(r.tof_valid[0] == 0 && close(r.m2[0], kNA), "unmatched arm stays missing");
    CHECK(r.tof_valid[1] == 0 && close(r.m2[1], kNA),
          "m2 == -1000 sentinel must be rejected, not stored");
    t.m2b = 0.88;
    r = FillTrackRow<FakeGlobal, FakeInner>(&t, 0, nullptr);
    CHECK(r.tof_valid[1] == 1 && close(r.m2[1], 0.88), "valid m2 is kept");
    CHECK(r.tof_valid[0] == 0, "the two arms stay independent");
  }

  // --- a null inner array leaves association unknown, never guessed -------
  {
    FakeGlobal t; t.par = {0.5, 0, 0}; t.inner = 3; t.primary = true;
    auto r = FillTrackRow<FakeGlobal, FakeInner>(&t, 0, nullptr);
    // IsPrimary lives on the global track, so it survives a missing inner
    // array; the inner-track columns do not and must stay unknown.
    CHECK(r.from_vertex == 1, "primary flag comes from the global track");
    CHECK(r.inner_hits == -1, "no inner array -> hit count unknown");
    CHECK(r.mc_index == -1, "no inner array -> no MC association");
  }

  // --- MC fields are populated but cannot reach a selection --------------
  {
    FakeInner inner{12, 42};
    TClonesArray arr; arr.v = {nullptr, nullptr, nullptr, &inner};
    FakeGlobal t; t.par = {0.5, 0, 0}; t.inner = 3; t.primary = true;
    auto r = FillTrackRow<FakeGlobal, FakeInner>(&t, 0, &arr);
    CHECK(r.from_vertex == 1 && r.inner_hits == 12, "inner track read");
    CHECK(r.mc_index == 42, "MC index recorded for validation");
    // every MC column in the schema must carry the mc_ prefix
    std::string h = kTracksHeader();
    for (const char* f : {"mc_index", "mc_pdg", "mc_mother"})
      CHECK(h.find(f) != std::string::npos, "MC column present and prefixed");
    CHECK(h.find(",pdg") == std::string::npos && h.find(",mother") == std::string::npos,
          "no unprefixed truth column may exist in the track schema");
  }

  // --- chi2/ndf guarded against ndf = 0 ----------------------------------
  {
    FakeGlobal t; t.par = {0.5, 0, 0}; t.chi2 = 10.0; t.ndf = 0;
    auto r = FillTrackRow<FakeGlobal, FakeInner>(&t, 0, nullptr);
    CHECK(close(r.chi2_ndf, kNA), "ndf=0 must not divide");
    t.ndf = 5;
    r = FillTrackRow<FakeGlobal, FakeInner>(&t, 0, nullptr);
    CHECK(close(r.chi2_ndf, 2.0), "chi2/ndf");
  }

  // --- source_id is stable, basename-only, and separates files -----------
  {
    long a = SourceId("/some/where/0001_dst.root");
    long b = SourceId("/elsewhere/0001_dst.root");
    long c = SourceId("/some/where/0002_dst.root");
    CHECK(a == b, "source_id must depend on the basename only");
    CHECK(a != c, "different files must get different source_id");
    CHECK(a > 0 && c > 0, "source_id must stay positive");
  }

  // --- a zero-track event still writes an event row ----------------------
  {
    CsvWriter w;
    CHECK(w.open("/tmp/bmnx_events_test.csv", kEventsHeader()), "open events csv");
    EventRow e;                       // all defaults: no tracks, no vertex
    e.event_id = 5;
    e.write(w, 1234, 7);
    w.close();
    FILE* f = fopen("/tmp/bmnx_events_test.csv", "r");
    char line[512]; int n = 0; std::string last;
    while (fgets(line, sizeof line, f)) { last = line; ++n; }
    fclose(f);
    CHECK(n == 2, "header plus exactly one row for a zero-track event");
    CHECK(last.rfind("1234,7,5,0,0,0,0,0,0", 0) == 0,
          "zero-track event row must still carry its keys and zero counts");
  }

  if (fails == 0) printf("all track-export checks passed\n");
  else printf("%d check(s) FAILED\n", fails);
  return fails ? 1 : 0;
}
