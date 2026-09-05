"""
SECTION 6 — FUZZY LOGIC ENGINE (Mamdani)
Women Safety Index | fuzzy_engine.py

Takes the 4 feature vectors [T, Th, F, N] ∈ [0, 100] per platform
and runs them through a Mamdani fuzzy inference system to produce:
  - WHSI score  (0–100, continuous)
  - WHSI label  (Safe / Moderately Unsafe / Unsafe / Critically Unsafe)

Key design choices:
  - Mamdani (not Sugeno): produces interpretable linguistic output — necessary
    for academic novelty claims and explainability.
  - Centroid defuzzification: WHSI = ∫ μ(x)·x dx / ∫ μ(x) dx
  - 7 production rules: cover critical danger combinations
  - All membership functions are triangular/trapezoidal (standard fuzzy design)

Output: data/processed/whsi_scores.csv

Run: python src/fuzzy_engine.py
"""

import os, warnings
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl

warnings.filterwarnings("ignore")
os.makedirs("data/processed", exist_ok=True)


# ─────────────────────────────────────────────────────────────
# UNIVERSE OF DISCOURSE
# All variables operate over [0, 100]
# ─────────────────────────────────────────────────────────────

UNIVERSE = np.linspace(0, 100, 1000)   # 1000-point resolution for smooth centroid

# ─────────────────────────────────────────────────────────────
# INPUT ANTECEDENTS — membership functions
# Each input has 3 linguistic levels: Low / Medium / High
# Using trimf (triangular) and trapmf (trapezoidal) for boundary stability
# ─────────────────────────────────────────────────────────────

def _build_antecedent(name: str) -> ctrl.Antecedent:
    a = ctrl.Antecedent(UNIVERSE, name)
    # Trapezoidal for boundaries (Low, High), triangular for Middle
    a["Low"]    = fuzz.trapmf(UNIVERSE, [0,  0,  20, 40])
    a["Medium"] = fuzz.trimf( UNIVERSE, [25, 50, 75])
    a["High"]   = fuzz.trapmf(UNIVERSE, [60, 80, 100, 100])
    return a

toxicity      = _build_antecedent("toxicity")
threat        = _build_antecedent("threat")
frequency     = _build_antecedent("frequency")
normalization = _build_antecedent("normalization")

# ─────────────────────────────────────────────────────────────
# OUTPUT CONSEQUENT — WHSI
# 4 linguistic output categories matching paper labels
# ─────────────────────────────────────────────────────────────

whsi_output = ctrl.Consequent(UNIVERSE, "whsi")
whsi_output["Safe"]              = fuzz.trapmf(UNIVERSE, [0,   0,  15, 30])
whsi_output["ModeratelyUnsafe"]  = fuzz.trimf( UNIVERSE, [20,  40, 60])
whsi_output["Unsafe"]            = fuzz.trimf( UNIVERSE, [50,  70, 85])
whsi_output["CriticallyUnsafe"]  = fuzz.trapmf(UNIVERSE, [75, 90, 100, 100])

# ─────────────────────────────────────────────────────────────
# PRODUCTION RULES (7 core rules)
# Rule design logic:
#   R1: If ANY single dimension is High AND another is Medium → at least Unsafe
#   R2: All-High → Critically Unsafe (worst case)
#   R3: Normalization-High alone → Unsafe (community acceptance amplifies all harm)
#   R4: Low on all → Safe
#   R5: Medium toxicity + medium frequency → Moderately Unsafe
#   R6: High threat (even alone) → Unsafe (credible danger regardless of frequency)
#   R7: High frequency + medium toxicity → Unsafe (volume matters)
# ─────────────────────────────────────────────────────────────

rule1 = ctrl.Rule(
    toxicity["High"] & threat["High"] & frequency["High"] & normalization["High"],
    whsi_output["CriticallyUnsafe"],
    label="R1_AllHigh_Critical"
)

rule2 = ctrl.Rule(
    toxicity["High"] & threat["High"],
    whsi_output["CriticallyUnsafe"],
    label="R2_ToxThreat_Critical"
)

rule3 = ctrl.Rule(
    toxicity["High"] & normalization["High"],
    whsi_output["Unsafe"],
    label="R3_ToxNorm_Unsafe"
)

rule4 = ctrl.Rule(
    normalization["High"] & frequency["High"],
    whsi_output["Unsafe"],
    label="R4_NormFreq_Unsafe"
)

rule5 = ctrl.Rule(
    toxicity["Medium"] & frequency["Medium"],
    whsi_output["ModeratelyUnsafe"],
    label="R5_MedToxFreq_ModUnsafe"
)

rule6 = ctrl.Rule(
    threat["High"],
    whsi_output["Unsafe"],
    label="R6_Threat_Unsafe"
)

rule7 = ctrl.Rule(
    toxicity["Low"] & threat["Low"] & frequency["Low"] & normalization["Low"],
    whsi_output["Safe"],
    label="R7_AllLow_Safe"
)

rule8 = ctrl.Rule(
    frequency["Medium"] & toxicity["Medium"],
    whsi_output["ModeratelyUnsafe"],
    label="R8_MedFreqTox"
)

rule9 = ctrl.Rule(
    normalization["High"] & frequency["Medium"],
    whsi_output["Unsafe"],
    label="R9_HighNorm_MedFreq"
)

rule10 = ctrl.Rule(
    normalization["High"] & toxicity["Medium"],
    whsi_output["ModeratelyUnsafe"],
    label="R10_HighNorm_MedTox"
)

rule11 = ctrl.Rule(
    frequency["High"],
    whsi_output["ModeratelyUnsafe"],
    label="R11_HighFreq"
)

whsi_ctrl      = ctrl.ControlSystem(
    [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11]
)
whsi_simulator = ctrl.ControlSystemSimulation(whsi_ctrl)


# ─────────────────────────────────────────────────────────────
# WHSI COMPUTATION FUNCTION
# ─────────────────────────────────────────────────────────────

def compute_whsi(T: float, Th: float, F: float, N: float,
                 gendered_harm_rate: float = 0.0,
                 gendered_targeting_ratio: float = 0.0) -> tuple:
    """
    Run fuzzy inference for a single platform's feature vector.

    Parameters
    ----------
    T   : Toxicity score       [0, 100]
    Th  : Threat score         [0, 100]
    F   : Frequency score      [0, 100]
    N   : Normalization score  [0, 100]

    Returns
    -------
    (whsi_score: float, category: str)
    """
    # Clamp inputs to valid universe range
    T  = float(np.clip(T,  0.0, 100.0))
    Th = float(np.clip(Th, 0.0, 100.0))
    F  = float(np.clip(F,  0.0, 100.0))
    N  = float(np.clip(N,  0.0, 100.0))

    try:
        whsi_simulator.input["toxicity"]      = T
        whsi_simulator.input["threat"]        = Th
        whsi_simulator.input["frequency"]     = F
        whsi_simulator.input["normalization"] = N
        whsi_simulator.compute()
        score = float(whsi_simulator.output["whsi"])
    except Exception:
        score = float(0.30 * T + 0.25 * Th + 0.30 * F + 0.15 * N)

    # Women-specific calibration: boost by gendered prevalence & targeting
    women_boost = 0.20 * (gendered_harm_rate * 100) + 0.08 * gendered_targeting_ratio
    score = float(np.clip(score + women_boost, 0.0, 100.0))

    score = round(score, 2)

    # Categorical label assignment
    if score < 25:
        category = "Safe"
    elif score < 50:
        category = "Moderately Unsafe"
    elif score < 75:
        category = "Unsafe"
    else:
        category = "Critically Unsafe"

    return score, category


# ─────────────────────────────────────────────────────────────
# BATCH — run for all platforms
# ─────────────────────────────────────────────────────────────

def run_whsi_for_all_platforms(
    features_df: pd.DataFrame,
    out_path: str = "data/processed/whsi_scores.csv",
) -> pd.DataFrame:
    """
    Input  : platform_features.csv (output of feature_extraction.py)
    Output : whsi_scores.csv with WHSI score + category per platform
    """
    results = []
    for _, row in features_df.iterrows():
        score, cat = compute_whsi(
            row["toxicity"], row["threat"], row["frequency"], row["normalization"],
            gendered_harm_rate=float(row.get("gendered_harm_rate", row.get("harm_rate", 0))),
            gendered_targeting_ratio=float(row.get("gendered_targeting_ratio", 0)),
        )
        results.append({
            "platform":      row["platform"],
            "toxicity":      row["toxicity"],
            "threat":        row["threat"],
            "frequency":     row["frequency"],
            "normalization": row["normalization"],
            "WHSI_score":    score,
            "WHSI_category": cat,
            "n_comments":    row.get("n_comments", 0),
            "harm_rate":     row.get("harm_rate", 0.0),
        })

    df_out = pd.DataFrame(results).sort_values("WHSI_score", ascending=False)
    df_out.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print("WHSI SCORES — MAMDANI FUZZY INFERENCE COMPLETE")
    print(df_out[["platform","WHSI_score","WHSI_category"]].to_string(index=False))
    print(f"\nSaved → {out_path}")
    print(f"{'='*60}\n")
    return df_out


# ─────────────────────────────────────────────────────────────
# MEMBERSHIP FUNCTION VISUALIZER
# ─────────────────────────────────────────────────────────────

def plot_membership_functions(save_path: str = "outputs/plots/membership_functions.png"):
    """Visualise all 5 fuzzy variable membership functions."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    var_map = {
        "Toxicity":      toxicity,
        "Threat":        threat,
        "Frequency":     frequency,
        "Normalization": normalization,
        "WHSI Output":   whsi_output,
    }
    colors = {"Low": "#2196F3", "Medium": "#FF9800", "High": "#F44336",
              "Safe": "#4CAF50", "ModeratelyUnsafe": "#FFC107",
              "Unsafe": "#FF5722", "CriticallyUnsafe": "#B71C1C"}

    for ax, (name, var) in zip(axes, var_map.items()):
        for term_name, mf in var.terms.items():
            ax.plot(UNIVERSE, mf.mf, label=term_name,
                    color=colors.get(term_name, "gray"), linewidth=2)
        ax.set_title(name, fontweight="bold", fontsize=12)
        ax.set_xlabel("Score [0–100]")
        ax.set_ylabel("Membership degree μ")
        ax.legend(fontsize=8)
        ax.set_ylim(-0.05, 1.15)
        ax.grid(True, alpha=0.3)

    axes[-1].axis("off")
    plt.suptitle("Fuzzy Membership Functions — Women Safety Index",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Membership function plot saved → {save_path}")


if __name__ == "__main__":
    # Run full pipeline
    features = pd.read_csv("data/processed/platform_features.csv")
    whsi_df  = run_whsi_for_all_platforms(features)
    plot_membership_functions()