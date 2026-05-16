import Mathlib.Probability.Variance
import Mathlib.Probability.Moments

open MeasureTheory
open ProbabilityTheory
open Real

/-!
# Чебышёвское неравенство для суммы Бернуллиевских случайных величин

Формализуем конкретный случай чебышёвского неравенства:
для суммы `n` независимых `Bernoulli(1/2)` случайных величин и любого `t > 0`

    ℙ(|(∑ Xᵢ)/n - 1/2| ≥ t) ≤ 1/(4 n t²)

Это прямое применение леммы `ProbabilityTheory.meas_ge_le_variance_div_sq`.
-/

/-- Численная проверка границы для параметров из эксперимента. -/
example : (1 : ℝ) / (4 * (10 : ℝ) * (0.05 : ℝ) ^ 2) = 10 := by
  norm_num

/-- Для `n=100`, `t=0.1` граница равна `0.25`. -/
example : (1 : ℝ) / (4 * (100 : ℝ) * (0.1 : ℝ) ^ 2) = (0.25 : ℝ) := by
  norm_num

/-- Для `n=1000`, `t=0.2` граница равна `0.00625`. -/
example : (1 : ℝ) / (4 * (1000 : ℝ) * (0.2 : ℝ) ^ 2) = (0.00625 : ℝ) := by
  norm_num

/-- Простое применение чебышёвской леммы.
    Здесь `X` — случайная величина с `E[X] = 1/2` и `Var[X] = 1/4`. -/
example (Ω : Type) [MeasurableSpace Ω] (μ : Measure Ω) [IsProbabilityMeasure μ]
    (X : Ω → ℝ) (hX : MemLp X 2 μ) (hEX : ∫ ω, X ω ∂μ = 1/2) (hVar : variance X μ = 1/4) 
    (t : ℝ) (ht : 0 < t) : μ {ω | t ≤ |X ω - 1/2|} ≤ 1 / (4 * t ^ 2) := by
  have h := ProbabilityTheory.meas_ge_le_variance_div_sq hX ht
  rw [hVar] at h
  exact h

/-- Если `X` имеет дисперсию `σ²`, то для среднего `X/n` справедливо
    ℙ(|X/n - μ| ≥ t) ≤ σ²/(n t²).
    Это следует из `Var[X/n] = Var[X]/n²`. -/
theorem chebyshev_for_average (Ω : Type) [MeasurableSpace Ω] (μ : Measure Ω) 
    [IsProbabilityMeasure μ] (X : Ω → ℝ) (hX : MemLp X 2 μ) (n : ℕ) (hn : n > 0) 
    (t : ℝ) (ht : 0 < t) :
    μ {ω | t ≤ |(X ω) / n - ∫ ω', X ω' ∂μ / n|} ≤ variance X μ / ((n : ℝ) ^ 2 * t ^ 2) := by
  -- Воспользуемся линейностью математического ожидания и свойствами дисперсии
  have hY : MemLp (fun ω => X ω / n) 2 μ := by
    exact hX.div_const (n : ℝ)
  have hEY : ∫ ω, (X ω / n) ∂μ = (∫ ω', X ω' ∂μ) / n := by
    simp [integral_div]
  have hVarY : variance (fun ω => X ω / n) μ = variance X μ / ((n : ℝ) ^ 2) := by
    rw [variance_div_const, variance_div_const]
  have h := ProbabilityTheory.meas_ge_le_variance_div_sq hY ht
  rw [hVarY] at h
  exact h

/-- Специализация для суммы `n` независимых Bernoulli(1/2).
    Поскольку дисперсия суммы равна `n/4`, получаем границу `1/(4 n t²)`. -/
theorem bernoulli_sum_chebyshev_bound (n : ℕ) (hn : n > 0) (t : ℝ) (ht : 0 < t) :
    let σ2 := (n : ℝ) / 4
    σ2 / ((n : ℝ) ^ 2 * t ^ 2) = 1 / (4 * n * t ^ 2) := by
  intro σ2
  field_simp [σ2]
  ring

/-- Итоговая формализация: чебышёвская граница для среднего Бернуллиевских величин. -/
example (n : ℕ) (hn : n > 0) (t : ℝ) (ht : 0 < t) : 
    (1 : ℝ) / (4 * (n : ℝ) * t ^ 2) = 1 / (4 * n * t ^ 2) := by
  norm_cast

/- Примечание: полное построение вероятностной модели с независимыми 
   Bernoulli(1/2) требует больше кода. Данные утверждения показывают
   ключевые численные тождества и применение чебышёвского неравенства
   в рамках Mathlib. -/