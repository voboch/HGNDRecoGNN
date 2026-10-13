// Standalone tests of the applied PID: domain rejection, missing-data handling,
// the slide equations, and that no MC quantity can reach a decision.
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
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
#include "../../ncx_refactor/BmnProtonPid.h"

static int fails = 0;
#define CHECK(c,m) do{ if(!(c)){ printf("  FAIL %s\n", m); ++fails; } }while(0)

static std::string WriteCalib(const char* path, bool complete = true,
                              bool with_version = true) {
  std::ofstream o(path);
  o << "# test calibration\n";
  if (with_version) {
    o << "version test-v1\n";
    o << "fitted_on SMASH defaultSpot development half, files 0-99\n";
  }
  o << "units dedx arb; m2 GeV^2; p GeV/c\n";
  o << "p_range 0.5 4.0\n";
  o << "tof_arm 400\n";
  o << "cuts 2.0 3.0\n";
  o << "proton_bethe 1.0 2.0 0.1 2.0 2.0\n";
  if (complete) o << "pion_bethe 1.0 2.0 0.1 2.0 2.0\n";
  o << "proton_sigma_rel_dedx 0.08\n";
  o << "proton_sigma_m2 0.05\n";
  o << "pion_sigma_rel_dedx 0.08\n";
  o << "pion_sigma_m2 0.05\n";
  o.close();
  return path;
}

int main() {
  using namespace bmnx;

  // --- an incomplete calibration must not load ---------------------------
  {
    auto c = LoadPidCalibration(WriteCalib("/tmp/pid_bad.txt", false));
    CHECK(!c.ok, "a calibration missing pion_bethe must not load");
    auto s = ScoreTrack(c, 2.0, 1.0, true, 0.88, true, +1);
    CHECK(s.status == PidStatus::kNoCalibration && !s.selected,
          "no calibration -> no score and no selection");
  }

  // --- an unattributable calibration must not load -----------------------
  {
    auto c = LoadPidCalibration(WriteCalib("/tmp/pid_nover.txt", true, false));
    CHECK(!c.ok, "a calibration without version/fitted_on must be refused");
  }

  auto c = LoadPidCalibration(WriteCalib("/tmp/pid_ok.txt"));
  CHECK(c.ok, "the complete calibration loads");
  CHECK(c.tof_arm == 400, "arm is explicit, never a union");
  CHECK(c.version == "test-v1", "version carried");
  CHECK(c.fitted_on.find("development") != std::string::npos,
        "the development sample is recorded with the calibration");

  // --- domain: outside the fitted range is rejected, not extrapolated ----
  {
    auto lo = ScoreTrack(c, 0.4, 1.0, true, 0.88, true, +1);
    auto hi = ScoreTrack(c, 4.1, 1.0, true, 0.88, true, +1);
    auto in = ScoreTrack(c, 2.0, 1.0, true, 0.88, true, +1);
    CHECK(lo.status == PidStatus::kOutOfDomain && !lo.selected, "below p_min rejected");
    CHECK(hi.status == PidStatus::kOutOfDomain && !hi.selected, "above p_max rejected");
    CHECK(in.scored(), "inside the domain is scored");
    // boundaries are inclusive and must behave
    CHECK(ScoreTrack(c, 0.5, 1.0, true, 0.88, true, +1).scored(), "p_min inclusive");
    CHECK(ScoreTrack(c, 4.0, 1.0, true, 0.88, true, +1).scored(), "p_max inclusive");
  }

  // --- missing detector information is explicit, never substituted -------
  {
    auto nod = ScoreTrack(c, 2.0, 0.0, false, 0.88, true, +1);
    auto not_ = ScoreTrack(c, 2.0, 1.0, true, kNA, false, +1);
    CHECK(nod.status == PidStatus::kNoDedx && !nod.selected, "missing dE/dx rejected");
    CHECK(not_.status == PidStatus::kNoTof && !not_.selected, "missing TOF rejected");
  }

  // --- negative tracks are never protons ---------------------------------
  {
    auto neg = ScoreTrack(c, 2.0, 1.0, true, 0.88, true, -1);
    auto unk = ScoreTrack(c, 2.0, 1.0, true, 0.88, true, 0);
    CHECK(neg.status == PidStatus::kBadCharge && !neg.selected, "negative rejected");
    CHECK(unk.status == PidStatus::kBadCharge && !unk.selected, "unknown sign rejected");
  }

  // --- the slide equations ------------------------------------------------
  {
    const double p = 2.0;
    const double fp = c.proton.dedx_expected(p);
    // a track sitting exactly on the proton curve with the proton mass-squared
    auto s = ScoreTrack(c, p, fp, true, kProtonMass * kProtonMass, true, +1);
    CHECK(std::fabs(s.x_p) < 1e-12, "x_p = 0 on the proton curve");
    CHECK(std::fabs(s.y_p) < 1e-12, "y_p = 0 at the proton mass");
    CHECK(std::fabs(s.r_p) < 1e-12, "r_p = 0");
    CHECK(s.selected, "a perfect proton passes r_p<2 and r_pi>3");
    // r is the quadrature sum of its parts
    auto t = ScoreTrack(c, p, fp * (1.0 + 0.08), true,
                        kProtonMass * kProtonMass + 0.05, true, +1);
    CHECK(std::fabs(t.x_p - 1.0) < 1e-9, "one relative sigma in dE/dx -> x_p = 1");
    CHECK(std::fabs(t.y_p - 1.0) < 1e-9, "one sigma in m2 -> y_p = 1");
    CHECK(std::fabs(t.r_p - std::sqrt(2.0)) < 1e-9, "r = hypot(x,y)");
  }

  // --- the cut is a 2D radius, not a 1D two-sigma ------------------------
  {
    const double p = 2.0, fp = c.proton.dedx_expected(p);
    // offset purely in dE/dx by 1.9 sigma: inside r_p<2
    auto a = ScoreTrack(c, p, fp * (1.0 + 0.08 * 1.9), true,
                        kProtonMass * kProtonMass, true, +1);
    CHECK(a.r_p < 2.0, "1.9 sigma in one coordinate is inside the radius");
    // 1.5 sigma in each coordinate: r = 2.12, outside, though neither
    // coordinate alone exceeds 2
    auto b = ScoreTrack(c, p, fp * (1.0 + 0.08 * 1.5), true,
                        kProtonMass * kProtonMass + 0.05 * 1.5, true, +1);
    CHECK(std::fabs(b.x_p - 1.5) < 1e-9 && std::fabs(b.y_p - 1.5) < 1e-9, "setup");
    CHECK(b.r_p > 2.0 && !b.selected,
          "1.5 sigma in BOTH coordinates must fail r_p<2, which a per-axis "
          "two-sigma cut would have kept");
  }

  // --- the pion veto actually vetoes -------------------------------------
  {
    const double p = 2.0, fpi = c.pion.dedx_expected(p);
    // a track on the PION curve at the pion mass: r_pi = 0, so r_pi > 3 fails
    auto s = ScoreTrack(c, p, fpi, true, kPionMass * kPionMass, true, +1);
    CHECK(std::fabs(s.r_pi) < 1e-12, "r_pi = 0 on the pion curve");
    CHECK(!s.selected, "a pion-like track must be vetoed even if r_p happened to pass");
  }

  // --- no MC quantity is reachable from a decision ------------------------
  {
    // ScoreTrack's signature carries only measured quantities. Assert that the
    // PID output schema exposes no truth column, so a downstream selection
    // cannot read one by accident either.
    std::string h = kPidHeader();
    for (const char* f : {"mc_", "pdg", "mother", "truth"})
      CHECK(h.find(f) == std::string::npos,
            "the PID table must expose no MC column");
    CHECK(h.find("calib_version") != std::string::npos,
          "every scored row carries its calibration provenance");
  }

  if (!fails) printf("all proton-PID checks passed\n");
  else printf("%d check(s) FAILED\n", fails);
  return fails ? 1 : 0;
}
