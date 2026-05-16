#!/usr/bin/env python3
"""
Симуляция неравенства Чебышева и Хёффдинга для среднего Бернуллиевских величин.

Вопрос: насколько плотно неравенство Чебышева для среднего n независимых
подбрасываний честной монеты (p=1/2).

Параметры:
    n_values = [10, 100, 1000, 10000]   # размер выборки
    t_values = [0.05, 0.1, 0.2]         # пороги отклонения
    num_trials = 10_000                  # число независимых экспериментов для каждого (n, t)

Выход:
    - results.json: словарь с эмпирическими вероятностями, границами Чебышева и Хёффдинга
    - plots/*.png: графики для визуального сравнения
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# --- параметры ---
SEED = 42
N_VALUES = [10, 100, 1000, 10000]
T_VALUES = [0.05, 0.1, 0.2]
NUM_TRIALS = 10_000

# --- подготовка директорий ---
experiment_dir = Path(__file__).parent
plots_dir = experiment_dir / "plots"
plots_dir.mkdir(exist_ok=True)

results_path = experiment_dir / "results.json"

# --- теоретические границы ---
def chebyshev_bound(n, t):
    """Верхняя оценка Чебышева: 1/(4 n t^2)"""
    return 1.0 / (4 * n * t**2)

def hoeffding_bound(n, t):
    """Верхняя оценка Хёффдинга: 2 * exp(-2 n t^2)"""
    return 2.0 * np.exp(-2 * n * t**2)

# --- симуляция ---
def run_simulation():
    np.random.seed(SEED)
    results = {
        "parameters": {
            "seed": SEED,
            "n_values": N_VALUES,
            "t_values": T_VALUES,
            "num_trials": NUM_TRIALS,
        },
        "data": {}
    }
    
    for n in N_VALUES:
        results["data"][str(n)] = {}
        for t in T_VALUES:
            # Генерация всех испытаний сразу (num_trials строк, n столбцов)
            # Каждая строка — одна независимая выборка размера n
            samples = np.random.binomial(1, 0.5, size=(NUM_TRIALS, n))
            means = samples.mean(axis=1)  # среднее по каждой выборке
            deviations = np.abs(means - 0.5)
            emp_prob = (deviations >= t).mean()
            
            results["data"][str(n)][str(t)] = {
                "empirical_probability": float(emp_prob),
                "chebyshev_bound": float(chebyshev_bound(n, t)),
                "hoeffding_bound": float(hoeffding_bound(n, t)),
            }
    
    # Сохранение сырых результатов
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Симуляция завершена. Результаты сохранены в {results_path}")
    return results

# --- построение графиков ---
def plot_results(results):
    """Построить графики для каждого t по всем n (логарифмическая шкала по y)"""
    data = results["data"]
    n_vals = N_VALUES
    
    for t in T_VALUES:
        t_str = str(t)
        emp_vals = []
        cheb_vals = []
        hoeff_vals = []
        
        for n in n_vals:
            d = data[str(n)][t_str]
            emp_vals.append(d["empirical_probability"])
            cheb_vals.append(d["chebyshev_bound"])
            hoeff_vals.append(d["hoeffding_bound"])
        
        plt.figure(figsize=(10, 6))
        plt.semilogy(n_vals, emp_vals, 'o-', label="Эмпирическая вероятность", linewidth=2)
        plt.semilogy(n_vals, cheb_vals, 's--', label="Чебышев (1/(4 n t²))", linewidth=2)
        plt.semilogy(n_vals, hoeff_vals, '^--', label="Хёффдинг (2·exp(-2 n t²))", linewidth=2)
        plt.xscale('log')
        plt.grid(True, which="both", linestyle='--', alpha=0.6)
        plt.xlabel("Размер выборки n", fontsize=12)
        plt.ylabel("Вероятность отклонения", fontsize=12)
        plt.title(f"Отклонение |mean - 1/2| ≥ {t} для Бернулли(p=1/2)", fontsize=14)
        plt.legend(fontsize=11)
        plt.tight_layout()
        
        # Сохранение
        plot_path = plots_dir / f"deviation_t_{t:.2f}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  График сохранён: {plot_path}")
    
    # Дополнительный график: сравнение границ при фиксированном n=100
    n_fixed = 100
    t_range = np.linspace(0.01, 0.3, 50)
    cheb_range = chebyshev_bound(n_fixed, t_range)
    hoeff_range = hoeffding_bound(n_fixed, t_range)
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(t_range, cheb_range, label=f"Чебышев (n={n_fixed})", linewidth=2)
    plt.semilogy(t_range, hoeff_range, label=f"Хёффдинг (n={n_fixed})", linewidth=2)
    plt.grid(True, which="both", linestyle='--', alpha=0.6)
    plt.xlabel("Порог t", fontsize=12)
    plt.ylabel("Верхняя граница вероятности", fontsize=12)
    plt.title(f"Сравнение границ при фиксированном n={n_fixed}", fontsize=14)
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    plot_path = plots_dir / f"bounds_vs_t_n_{n_fixed}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  График сохранён: {plot_path}")

# --- вывод таблицы с результатами ---
def print_summary(results):
    print("\n" + "="*80)
    print("Сводка результатов (вероятность отклонения |mean - 1/2| ≥ t)")
    print("="*80)
    data = results["data"]
    
    for n in N_VALUES:
        print(f"\nn = {n}:")
        print("  t | Эмпирич. | Чебышев   | Хёффдинг  | Ч/Эмпир. | Х/Эмпир.")
        print("  " + "-"*70)
        for t in T_VALUES:
            d = data[str(n)][str(t)]
            emp = d["empirical_probability"]
            cheb = d["chebyshev_bound"]
            hoeff = d["hoeffding_bound"]
            ratio_cheb = cheb / emp if emp > 0 else float('inf')
            ratio_hoeff = hoeff / emp if emp > 0 else float('inf')
            print(f"  {t:.2f} | {emp:.2e} | {cheb:.2e} | {hoeff:.2e} |"
                  f" {ratio_cheb:.1f}    | {ratio_hoeff:.1f}")

# --- главная функция ---
if __name__ == "__main__":
    print("Запуск симуляции неравенств Чебышева и Хёффдинга...")
    print(f"Параметры: n = {N_VALUES}, t = {T_VALUES}, число испытаний = {NUM_TRIALS}")
    
    results = run_simulation()
    plot_results(results)
    print_summary(results)
    
    print("\n" + "="*80)
    print("Этап 2 (численный эксперимент) завершён.")
    print(f"Результаты: {results_path}")
    print(f"Графики: {plots_dir}/")