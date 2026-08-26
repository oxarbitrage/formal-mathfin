/-
Copyright (c) 2026 Raphael Coelho. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Raphael Coelho
-/
module

public import MathFin.Foundations.ItoIntegralAgainstMartingale
public import MathFin.Foundations.GaussianMoments

/-! # The pointwise bracket and the conditional second moment

`bracketMeasure` (`ItoIntegralAgainstMartingale`) weights time-and-sample by `φ²`, which
integrates `ω` out: it can record *unconditional* second moments only. The bracket of the
standard theory is a *process*, `⟨M⟩_t(ω) = ∫₀ᵗ φ_s(ω)² ds`, and this file builds exactly that
much of it — enough for the conditional identity the measure-level object cannot state:

  `μ[(M_b − M_a)² | 𝓕_a] = μ[fun ω ↦ ∫ u in a..b, (⇑φ(u,ω))² du | 𝓕_a]`,

for `M = φ●B` on `[0,T]` and `a ≤ b ≤ T`. This is the defining property quadratic variation
is *for*, now conditionally — what `norm_sq_increment_eq_bracket` delivered unconditionally.

## Route

Both sides are quadratic in the integrand, so neither extends along a density linearly.
Instead both are polarised into the bilinear form (continuous `L²×L² → L¹`)

  `B(χ,η) := μ[Jχ·Jη | 𝓕_a] − μ[fun ω ↦ ∫ u in a..T, ⇑χ(u,ω)·⇑η(u,ω) du | 𝓕_a]`,

where `J = itoIntegralCLM_T`:

* **Kernels** — functions of a Brownian increment condition on the past as constants:
  `μ[B_v − B_u | 𝓕_u] =ᵐ 0`, `μ[(B_v−B_u)² | 𝓕_u] =ᵐ v−u`; tower + pull-out upgrade these to
  `𝓕_a`-adapted coefficients (`condExp_adapted_mul_increment_sq`, `condExp_adapted_mul_increment_zero`).
* **Generators** — single bands `Z·1_{(c,d]}` with `a ≤ c`: their integrals evaluate to the
  explicit increments `Z(B_d − B_c)` (`eval_bandIntegrand`), and the pair identity
  `B(gen, gen') = 0` (`condExp_pair_bands`) is a case analysis on the order of the two
  intervals, every case closed by the kernels plus increment splitting.
* **Density** — the span of post-`a` generators is dense in the post-`a`-supported part of
  `L²(trim_T)` (`mem_closure_postA_span`): a class orthogonal to all of them, clipped to
  `{t > a}`, kills every global predictable rectangle (filtration monotonicity makes
  `(c∨a, d] × F` legal again), so `setIntegral_eq_zero_of_orthogonal_pred` finishes it.
* **Extension** — ε/limit along that density: both sides converge in `L¹`, giving
  `μ[(Jψ)² | 𝓕_a] = μ[ωise ∫ (⇑ψ)² du | 𝓕_a]` for every post-`a`-supported `ψ`
  (`condExp_second_moment_postA`).

Applying it to `ψ = restrictAfterCLM a φ` reads off `M_T − M_a`; applying it to
`ψ = bandRestrict a b φ` and using `itoProcessCLM_eq_zero_of_vanishes_before` reads off the
band form above (`condExp_band_second_moment`).

## Honest scope

The bracket process is delivered through its increments' conditional expectations. This file
does **not** package `t ↦ ∫₀ᵗ φ_s² ds` as an *adapted increasing process* (predictability of
the representative does not give progressive measurability at this pin, and no pathwise
quadratic variation is constructed); what is named is the ω-wise integral of the squared
representative (`bracketRep`), with its monotonicity.
-/

@[expose] public section

namespace MathFin

open MeasureTheory ProbabilityTheory Filter Topology NNReal ENNReal
open ItoIntegralL2 ItoIntegralCLM ItoIsometryAdapted
open ItoIntegralAgainstMartingale LpMulIsometry ItoIntegralBrownian

namespace PointwiseBracket

variable {Ω : Type*} [mΩ : MeasurableSpace Ω] {μ : Measure Ω} [IsProbabilityMeasure μ]
  {B : ℝ≥0 → Ω → ℝ}

/-- Trims of the natural Brownian filtration are σ-finite: `μ` is a probability measure,
so its restriction to any sub-σ-algebra is finite, hence σ-finite. Side-condition shim for
the tower property of the conditional expectation (`condExp_condExp_of_le`). -/
private instance instSigmaFiniteTrimNatFiltration {hBmeas : ∀ t, Measurable (B t)}
    {u : ℝ≥0} : SigmaFinite (μ.trim ((natFiltration hBmeas).le u)) :=
  haveI : IsFiniteMeasure (μ.trim ((natFiltration hBmeas).le u)) :=
    MeasureTheory.isFiniteMeasure_trim _
  inferInstance

/-- For `s ≤ t : ℝ≥0`, the truncated increment variance `max (t-s) (s-t)` is the
`ℝ≥0`-nndistance of the coerced times. -/
private lemma maxSub_eq_nndist {s t : ℝ≥0} (hst : s ≤ t) :
    (max (t - s) (s - t) : ℝ≥0) = nndist (t : ℝ) (s : ℝ) := by
  apply NNReal.coe_injective
  have hle : ((s : ℝ) ≤ (t : ℝ)) := by exact_mod_cast hst
  rw [coe_nndist, Real.dist_eq, tsub_eq_zero_of_le hst, max_eq_left zero_le,
    NNReal.coe_sub hst, abs_of_nonneg (sub_nonneg.mpr hle)]

/-- For `s ≤ t : ℝ≥0`, the truncated variance coerces to the real elapsed time. -/
private lemma maxSubCoe {s t : ℝ≥0} (hst : s ≤ t) :
    ((max (t - s) (s - t) : ℝ≥0) : ℝ) = (t : ℝ) - (s : ℝ) := by
  have hst_zero : s - t = (0 : ℝ≥0) := tsub_eq_zero_of_le hst
  rw [hst_zero, max_eq_left zero_le]
  exact NNReal.coe_sub hst

/-! ### The conditional Brownian kernels -/

/-- Functions of a Brownian increment condition on the past as constants: if
`∫ φ(B_v − B_u) dμ = c` then `μ[φ(B_v − B_u) | 𝓕_u] =ᵐ c`. Independence of the increment
from `𝓕_u` (`IsFilteredPreBrownian.indep`) via `condExp_indep_eq`. -/
private theorem condExp_func_increment (hB : IsPreBrownianReal B μ)
    (hBmeas : ∀ t, Measurable (B t)) {u v : ℝ≥0} (huv : u ≤ v)
    {φ : ℝ → ℝ} (hφ : Measurable φ) {c : ℝ}
    (h_int : ∫ ω, φ (B v ω - B u ω) ∂μ = c) :
    (μ[fun ω ↦ φ (B v ω - B u ω) | natFiltration hBmeas u]) =ᵐ[μ] fun _ ↦ c := by
  have hd : Measurable (fun ω ↦ B v ω - B u ω) := (hBmeas v).sub (hBmeas u)
  have hcomp : Measurable[
      MeasurableSpace.comap (fun ω ↦ B v ω - B u ω) (borel ℝ)]
      (fun ω ↦ φ (B v ω - B u ω)) :=
    hφ.comp (Measurable.of_comap_le le_rfl)
  obtain ⟨hFB⟩ : Nonempty (IsFilteredPreBrownian B (natFiltration hBmeas) μ) :=
    ⟨hB.isFilteredPreBrownian hBmeas⟩
  have hindep := condExp_indep_eq hd.comap_le ((natFiltration hBmeas).le u)
    hcomp.stronglyMeasurable (hFB.indep u v huv)
  rwa [h_int] at hindep

omit [IsProbabilityMeasure μ] in
/-- **Kernel 1**: the increment has zero conditional mean — `μ[B_v − B_u | 𝓕_u] =ᵐ 0`. -/
theorem condExp_increment_eq_zero (hB : IsPreBrownianReal B μ)
    (hBmeas : ∀ t, Measurable (B t)) {u v : ℝ≥0} (huv : u ≤ v) :
    (μ[fun ω ↦ B v ω - B u ω | natFiltration hBmeas u]) =ᵐ[μ] fun _ ↦ (0 : ℝ) := by
  haveI : IsProbabilityMeasure μ := hB.isGaussianProcess.isProbabilityMeasure
  have hL : HasLaw (B v - B u)
      (gaussianReal 0 (max (v - u) (u - v))) μ := by
    rw [maxSub_eq_nndist huv]; exact hB.hasLaw_sub v u
  have hint : ∫ ω, (B v ω - B u ω) ∂μ = 0 := by
    have h_eq : (fun ω ↦ B v ω - B u ω) = (B v - B u : Ω → ℝ) := rfl
    rw [h_eq, hL.integral_eq, integral_id_gaussianReal]
  exact condExp_func_increment hB hBmeas huv measurable_id hint

omit [IsProbabilityMeasure μ] in
/-- **Kernel 2**: the squared increment conditions on the elapsed time —
`μ[(B_v − B_u)² | 𝓕_u] =ᵐ v − u`. -/
theorem condExp_increment_sq (hB : IsPreBrownianReal B μ)
    (hBmeas : ∀ t, Measurable (B t)) {u v : ℝ≥0} (huv : u ≤ v) :
    (μ[fun ω ↦ (B v ω - B u ω) ^ 2 | natFiltration hBmeas u])
      =ᵐ[μ] fun _ ↦ (v : ℝ) - u := by
  haveI : IsProbabilityMeasure μ := hB.isGaussianProcess.isProbabilityMeasure
  have hL : HasLaw (B v - B u)
      (gaussianReal 0 (max (v - u) (u - v))) μ := by
    rw [maxSub_eq_nndist huv]; exact hB.hasLaw_sub v u
  have hint : ∫ ω, (B v ω - B u ω) ^ 2 ∂μ = (v : ℝ) - u := by
    have h_change : ∫ ω, (B v ω - B u ω) ^ 2 ∂μ
        = ∫ x, x ^ 2 ∂(gaussianReal 0 (max (v - u) (u - v))) := by
      simpa [Function.comp] using hL.integral_comp (f := fun x : ℝ ↦ x ^ 2) (by fun_prop)
    rw [h_change, integral_sq_gaussianReal]
    exact maxSubCoe huv
  exact condExp_func_increment hB hBmeas huv (measurable_id.pow_const 2) hint

omit [IsProbabilityMeasure μ] in
/-- A Brownian increment lies in `L²(μ)`. -/
private theorem memLp_increment (hB : IsPreBrownianReal B μ)
    (hBmeas : ∀ t, Measurable (B t)) {u v : ℝ≥0} (huv : u ≤ v) :
    MemLp (fun ω ↦ B v ω - B u ω) 2 μ := by
  haveI : IsProbabilityMeasure μ := hB.isGaussianProcess.isProbabilityMeasure
  have hL : HasLaw (B v - B u)
      (gaussianReal 0 (max (v - u) (u - v))) μ := by
    rw [maxSub_eq_nndist huv]; exact hB.hasLaw_sub v u
  have hd : Measurable (fun ω ↦ B v ω - B u ω) := (hBmeas v).sub (hBmeas u)
  rw [show (fun ω ↦ B v ω - B u ω) = (B v - B u : Ω → ℝ) from rfl]
  exact ((hL.map_eq ▸ memLp_id_gaussianReal 2 :
    MemLp (id : ℝ → ℝ) 2 (Measure.map (B v - B u) μ))).comp_of_map hd.aemeasurable

omit [IsProbabilityMeasure μ] in
/-- **Kernel 3 (adapted diagonal)**: an `𝓕_c`-measurable coefficient times a squared
increment conditions on the elapsed time —
`μ[χ·(B_d−B_c)² | 𝓕_a] =ᵐ (d−c) · μ[χ | 𝓕_a]` for `a ≤ c ≤ d`. Tower through `𝓕_c`,
pull-out, Kernel 2; the product-integrability hypothesis is discharged by independence at
the call sites (`Indep.integrable_mul`). -/
theorem condExp_adapted_mul_increment_sq (hB : IsPreBrownianReal B μ)
    (hBmeas : ∀ t, Measurable (B t)) (_hprob : IsProbabilityMeasure μ)
    {a c d : ℝ≥0} (hac : a ≤ c) (hcd : c ≤ d)
    {χ : Ω → ℝ} (hχm : StronglyMeasurable[natFiltration hBmeas c] χ)
    (hχi : Integrable χ μ)
    (hint : Integrable (fun ω ↦ χ ω * (B d ω - B c ω) ^ 2) μ) :
    (μ[fun ω ↦ χ ω * (B d ω - B c ω) ^ 2 | natFiltration hBmeas a])
      =ᵐ[μ] fun ω ↦ ((d : ℝ) - c) * (μ[χ | natFiltration hBmeas a]) ω := by
  have hsq_int : Integrable (fun ω ↦ (B d ω - B c ω) ^ 2) μ :=
    (memLp_increment hB hBmeas hcd).integrable_sq
  have hinner : (μ[fun ω ↦ χ ω * (B d ω - B c ω) ^ 2 | natFiltration hBmeas c])
      =ᵐ[μ] fun ω ↦ χ ω * ((d : ℝ) - c) :=
    (condExp_mul_of_stronglyMeasurable_left hχm hint hsq_int).trans
      (by filter_upwards [condExp_increment_sq hB hBmeas hcd] with ω hω
          exact congrArg (χ ω * ·) hω)
  have hint' : Integrable (fun ω ↦ χ ω * ((d : ℝ) - c)) μ :=
    integrable_condExp.congr hinner
  calc (μ[fun ω ↦ χ ω * (B d ω - B c ω) ^ 2 | natFiltration hBmeas a])
      =ᵐ[μ] (μ[μ[fun ω ↦ χ ω * (B d ω - B c ω) ^ 2 | natFiltration hBmeas c]
          | natFiltration hBmeas a]) :=
        (condExp_condExp_of_le ((natFiltration hBmeas).mono hac)
          ((natFiltration hBmeas).le c)).symm
    _ =ᵐ[μ] (μ[(fun ω ↦ χ ω * ((d : ℝ) - c)) | natFiltration hBmeas a]) :=
        condExp_congr_ae hinner
    _ =ᵐ[μ] fun ω ↦ ((d : ℝ) - c) * (μ[χ | natFiltration hBmeas a]) ω :=
        (condExp_mul_of_aestronglyMeasurable_right
          stronglyMeasurable_const.aestronglyMeasurable hint' hχi).trans
          (Filter.Eventually.of_forall fun ω ↦ mul_comm _ _)

omit [IsProbabilityMeasure μ] in
/-- **Kernel 4 (adapted cross term)**: an `𝓕_u`-measurable coefficient times an increment
after `u` has zero conditional expectation given the further past —
`μ[X·(B_v−B_u) | 𝓕_a] =ᵐ 0` for `a ≤ u ≤ v`. Tower through `𝓕_u`, pull-out, Kernel 1. -/
theorem condExp_adapted_mul_increment_zero (hB : IsPreBrownianReal B μ)
    (hBmeas : ∀ t, Measurable (B t)) (_hprob : IsProbabilityMeasure μ)
    {a u v : ℝ≥0} (hau : a ≤ u) (huv : u ≤ v)
    {X : Ω → ℝ} (hXm : StronglyMeasurable[natFiltration hBmeas u] X)
    (hint : Integrable (fun ω ↦ X ω * (B v ω - B u ω)) μ) :
    (μ[fun ω ↦ X ω * (B v ω - B u ω) | natFiltration hBmeas a]) =ᵐ[μ] fun _ ↦ 0 := by
  have hinner : (μ[fun ω ↦ X ω * (B v ω - B u ω) | natFiltration hBmeas u])
      =ᵐ[μ] fun _ ↦ (0 : ℝ) :=
    (condExp_mul_of_stronglyMeasurable_left hXm hint
      ((memLp_increment hB hBmeas huv).integrable one_le_two)).trans
      (by filter_upwards [condExp_increment_eq_zero hB hBmeas huv] with ω hω
          show (X * μ[fun ω ↦ B v ω - B u ω | natFiltration hBmeas u]) ω = 0
          rw [Pi.mul_apply, hω, mul_zero])
  have houter : (μ[μ[fun ω ↦ X ω * (B v ω - B u ω) | natFiltration hBmeas u]
      | natFiltration hBmeas a]) =ᵐ[μ] fun _ ↦ (0 : ℝ) := by
    refine (condExp_congr_ae hinner).trans ?_
    exact Filter.EventuallyEq.of_eq condExp_zero
  calc (μ[fun ω ↦ X ω * (B v ω - B u ω) | natFiltration hBmeas a])
      =ᵐ[μ] (μ[μ[fun ω ↦ X ω * (B v ω - B u ω) | natFiltration hBmeas u]
          | natFiltration hBmeas a]) :=
        (condExp_condExp_of_le ((natFiltration hBmeas).mono hau)
          ((natFiltration hBmeas).le u)).symm
    _ =ᵐ[μ] fun _ ↦ (0 : ℝ) := houter


theorem coeFn_bandAssembly {T : ℝ≥0} (hBmeas : ∀ t, Measurable (B t))
    {c d : ℝ≥0} (hcd : c ≤ d) (hdT : d ≤ T)
    {Z : Ω → ℝ} (hZm : Measurable[natFiltration hBmeas c] Z) {C : ℝ}
    (hZb : ∀ ω, |Z ω| ≤ C) :
    ⇑(simpleAssembly_T (μ := μ) T hBmeas (stepSP hBmeas hcd hdT hZm hZb))
      =ᵐ[trimMeasure_T (μ := μ) T hBmeas]
        fun p ↦ (Set.Ioc c d).indicator (fun _ ↦ (1 : ℝ)) p.1 * Z p.2 := by
  refine (MemLp.coeFn_toLp (memLp_uncurry_trim_T T hBmeas
    (stepSP hBmeas hcd hdT hZm hZb).val)).trans
    (Filter.Eventually.of_forall fun p ↦ ?_)
  obtain ⟨t, ω⟩ := p
  show ⇑(stepSP hBmeas hcd hdT hZm hZb).val t ω
      = (Set.Ioc c d).indicator (fun _ ↦ (1 : ℝ)) t * Z ω
  rw [SimpleProcess.apply_eq]
  have hb0 : (stepSP hBmeas hcd hdT hZm hZb).val.valueBot = fun _ ↦ (0 : ℝ) := rfl
  have hbot : ({⊥} : Set ℝ≥0).indicator
      (fun _ ↦ (stepSP hBmeas hcd hdT hZm hZb).val.valueBot ω) t = 0 := by
    rw [hb0]
    by_cases h : t = ⊥ <;> simp [h]
  rw [hbot, zero_add,
    show (stepSP hBmeas hcd hdT hZm hZb).val.value
        = Finsupp.single (c, d) Z from rfl,
    Finsupp.sum_single_index (by simp)]
  by_cases hm : t ∈ Set.Ioc c d
  · rw [Set.indicator_of_mem hm, Set.indicator_of_mem hm, one_mul]
  · rw [Set.indicator_of_notMem hm, Set.indicator_of_notMem hm, zero_mul]

/-- **The single-band generator** `Z·1_{(c,d]×Ω}`, as a predictable `L²(trim_T)` class:
the assembly of the single-step process. For `a ≤ c`, these span the post-`a`-supported
part of `L²` (`dense_postA_span`), their integrals evaluate to the explicit increments
(`eval_bandGen`), and their pairwise conditional second moments vanish against each other
except through the time-overlap (`condExp_pair_bands`). -/
noncomputable def bandGen (T : ℝ≥0) (hBmeas : ∀ t, Measurable (B t))
    {c d : ℝ≥0} (hcd : c ≤ d) (hdT : d ≤ T)
    {Z : Ω → ℝ} (hZm : Measurable[natFiltration hBmeas c] Z) (C : ℝ)
    (hZb : ∀ ω, |Z ω| ≤ C) :
    Lp ℝ 2 (trimMeasure_T (μ := μ) T hBmeas) :=
  simpleAssembly_T (μ := μ) T hBmeas (stepSP hBmeas hcd hdT hZm hZb)

/-- Band generators vanish before their left endpoint. -/
theorem bandGen_support {T : ℝ≥0} (hBmeas : ∀ t, Measurable (B t))
    {c d : ℝ≥0} (hcd : c ≤ d) (hdT : d ≤ T)
    {Z : Ω → ℝ} (hZm : Measurable[natFiltration hBmeas c] Z) (C : ℝ)
    (hZb : ∀ ω, |Z ω| ≤ C) :
    ∀ᵐ p ∂(trimMeasure_T (μ := μ) T hBmeas), p.1 ≤ c →
      (bandGen (μ := μ) T hBmeas hcd hdT hZm C hZb : ℝ≥0 × Ω → ℝ) p = 0 := by
    filter_upwards [coeFn_bandAssembly (T := T) hBmeas hcd hdT hZm hZb] with p hcoe hpc
    show ⇑(simpleAssembly_T (μ := μ) T hBmeas (stepSP hBmeas hcd hdT hZm hZb)) p = 0
    rw [hcoe]
    show (Set.Ioc c d).indicator (fun _ ↦ (1 : ℝ)) p.1 * Z p.2 = 0
    rw [Set.indicator_of_notMem (fun hc ↦ absurd hc.1 (not_lt.mpr hpc)), zero_mul]

/-- **The band generator's integral is the explicit increment**: integrating
`Z·1_{(c,d]} dB` over `[0,T]` returns `Z·(B_d − B_c)` — the Riemann–Stieltjes term. -/
theorem eval_bandGen (hB : IsPreBrownianReal B μ) (T : ℝ≥0) (hBmeas : ∀ t, Measurable (B t))
    {c d : ℝ≥0} (hcd : c ≤ d) (hdT : d ≤ T)
    {Z : Ω → ℝ} (hZm : Measurable[natFiltration hBmeas c] Z) (C : ℝ)
    (hZb : ∀ ω, |Z ω| ≤ C) :
    ⇑(itoIntegralCLM_T hB T hBmeas (bandGen (μ := μ) T hBmeas hcd hdT hZm C hZb))
      =ᵐ[μ] fun ω ↦ Z ω * (B d ω - B c ω) := by
  rw [bandGen, itoIntegralCLM_T_simpleAssembly_T hB T hBmeas
    (stepSP hBmeas hcd hdT hZm hZb)]
  refine (MemLp.coeFn_toLp (memLp_itoSimple hB hBmeas
    (stepSP hBmeas hcd hdT hZm hZb).val)).trans
    (Filter.Eventually.of_forall fun ω ↦ ?_)
  rw [itoSimple_stepSP hBmeas hcd hdT hZm hZb]

end PointwiseBracket
end MathFin
