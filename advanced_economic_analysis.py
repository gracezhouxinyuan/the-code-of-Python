

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import SplineTransformer, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings("ignore")

from plot_style_utils import set_chinese_font

# 配色（按策略语义统一）
COLORS = {
    "C": "#2c3e50",
    "D": "#f3a69d",
    "B": "#e74c3c",
    "U": "#8fa1b3",
    "grid": "#8fa1b3",
}

plt.rcParams.update(
    {
        "font.family": "PingFang SC",
        "font.sans-serif": ["PingFang SC", "PingFang HK", "Arial Unicode MS"],
        "axes.edgecolor": COLORS["C"],
        "axes.labelcolor": COLORS["C"],
        "grid.color": COLORS["grid"],
        "grid.linestyle": "--",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "font.size": 10,
    }
)
sns.set_style("whitegrid")
set_chinese_font()

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_PATH = PROJECT_ROOT / "01 cleaned" / "gam_risk_model.pkl"
DISEASES_14 = [
    "Hypertension", "Dyslipidemia", "Diabetes", "Cancer", "Chronic lung disease",
    "Liver disease", "Heart disease", "Stroke", "Kidney disease", "Stomach disease",
    "Emotional problems", "Memory-related disease", "Arthritis", "Asthma",
]


def _thousands_formatter(x: float, _pos: int) -> str:
    return f"{x:,.0f}"


def _gamma_from_mean_sd(mean: float, sd: float, size: int) -> np.ndarray:
    shape = (mean / sd) ** 2
    scale = (sd**2) / mean
    return np.random.gamma(shape=shape, scale=scale, size=size)


def _beta_from_mean_sd(mean: float, sd: float, size: int) -> np.ndarray:
    var = sd**2
    common = mean * (1 - mean) / var - 1
    alpha = mean * common
    beta = (1 - mean) * common
    return np.random.beta(alpha, beta, size=size)


def _lognormal_from_mean_sd(mean: float, sd: float, size: int) -> np.ndarray:
    sigma2 = np.log(1 + (sd**2 / mean**2))
    sigma = np.sqrt(sigma2)
    mu = np.log(mean) - sigma2 / 2
    return np.random.lognormal(mean=mu, sigma=sigma, size=size)


def _safe_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce')
    if s.isna().all():
        return pd.Series(default, index=series.index)
    median_val = s.median() if s.notna().any() else default
    return s.fillna(median_val)


def _safe_cat(series: pd.Series, default: str = "unknown") -> pd.Series:
    s = series.astype(str).fillna(default)
    return s


def _build_any14_target(df: pd.DataFrame) -> pd.Series:
    if "any14_disease" in df.columns:
        return pd.to_numeric(df["any14_disease"], errors="coerce").fillna(0).astype(int)
    present = [c for c in DISEASES_14 if c in df.columns]
    if present:
        block = df[present].apply(pd.to_numeric, errors="coerce").fillna(0)
        return (block.max(axis=1) >= 1).astype(int)
    if "any_core4_disease" in df.columns:
        return pd.to_numeric(df["any_core4_disease"], errors="coerce").fillna(0).astype(int)
    htn = _safe_numeric(df.get("hypertension", 0))
    dm = _safe_numeric(df.get("diabetes", 0))
    return ((htn == 1) | (dm == 1)).astype(int)


def _prepare_features(df: pd.DataFrame):
    """准备模型所需的特征，确保列存在。"""
    df_model = df.copy()
    y = _build_any14_target(df_model)
    
    features = ['age', 'bmi', 'comorbidity_count', 'gender', 'smoking_status', 'drinking']
    for col in features:
        if col not in df_model.columns:
            if col == 'comorbidity_count':
                df_model[col] = 0
            elif col == 'bmi':
                df_model[col] = 24.0
            elif col == 'age':
                df_model[col] = 60.0
            elif col in ['gender', 'smoking_status', 'drinking']:
                df_model[col] = 'unknown'
    
    X = df_model[features].copy()
    for col in ['age', 'bmi', 'comorbidity_count']:
        X[col] = _safe_numeric(X[col], default=0)
    for col in ['gender', 'smoking_status', 'drinking']:
        X[col] = _safe_cat(X[col])
    for col in ['age', 'bmi', 'comorbidity_count']:
        X[col] = X[col].replace([np.inf, -np.inf], X[col].median() if X[col].notna().any() else 0)
    return X, y


def _get_risk_predictor(df: pd.DataFrame, force_retrain: bool = False):
    """训练或加载GAM风险预测模型，返回预测概率函数。"""
    if not force_retrain and MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        return lambda X: model.predict_proba(X)[:, 1]
    
    X, y = _prepare_features(df)
    numeric = ['age', 'bmi', 'comorbidity_count']
    categorical = ['gender', 'smoking_status', 'drinking']
    
    spline_trans = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('spline', SplineTransformer(n_knots=5, degree=3, include_bias=False)),
        ('scaler', StandardScaler(with_mean=False))
    ])
    other_trans = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_trans = Pipeline([
        ('imp', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ('spline', spline_trans, ['age', 'bmi']),
        ('num_other', other_trans, ['comorbidity_count']),
        ('cat', cat_trans, categorical)
    ])
    
    model = Pipeline([
        ('pre', preprocessor),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42))
    ])
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return lambda X: model.predict_proba(X)[:, 1]


def _calc_strategy_outcome(df: pd.DataFrame, strategy: str, p: dict, risk_predictor) -> tuple[float, float, float]:
    """返回 (total_cost, total_qaly, icer_vs_baseline)。"""
    n = len(df)
    true_cases = df["target"].values.astype(float)
    pred_prob = np.asarray(risk_predictor(df), dtype=float).reshape(-1)
    high_risk = (pred_prob >= np.quantile(pred_prob, 0.6)).astype(float)
    
    if strategy == "A":
        screened = np.zeros(n)
        intervention = np.zeros(n)
    elif strategy == "B":
        screened = (np.random.random(n) < 0.4).astype(float)
        intervention = np.zeros(n)
    elif strategy == "C":
        screened = high_risk
        intervention = np.zeros(n)
    elif strategy == "D":
        screened = high_risk
        intervention = high_risk
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    detected = true_cases * screened * p["screening_sensitivity"]
    qaly_gain = detected * p["qaly_per_case"] * (1 + intervention * (p["intervention_effect"] - 1))
    
    total_cost = (
        screened.sum() * p["screening_cost"]
        + detected.sum() * p["disease_management_cost"]
        + (detected * intervention).sum() * p["intervention_cost"]
    )
    total_qaly = qaly_gain.sum()
    icer = total_cost / total_qaly if total_qaly > 0 else np.nan
    return float(total_cost), float(total_qaly), float(icer)


def one_way_sensitivity_analysis(df: pd.DataFrame, risk_predictor) -> dict:
    print("\n" + "=" * 70)
    print("1. 单因素敏感性分析")
    print("=" * 70)
    
    base = {
        "screening_cost": 50.0,
        "intervention_cost": 600.0,
        "disease_management_cost": 1200.0,
        "screening_sensitivity": 0.72,
        "qaly_per_case": 0.20,
        "intervention_effect": 1.25,
    }
    
    perturbations = {
        "screening_cost": np.linspace(30, 80, 9),
        "intervention_cost": np.linspace(400, 900, 9),
        "screening_sensitivity": np.linspace(0.55, 0.85, 9),
        "qaly_per_case": np.linspace(0.12, 0.30, 9),
        "intervention_effect": np.linspace(1.05, 1.45, 9),
    }
    
    results = {}
    results["__baseline_icer__"] = float(_calc_strategy_outcome(df, "C", base, risk_predictor)[2])
    for k, values in perturbations.items():
        icers = []
        for v in values:
            p = base.copy()
            p[k] = float(v)
            _, _, icer_c = _calc_strategy_outcome(df, "C", p, risk_predictor)
            icers.append(icer_c)
        arr = np.array(icers, dtype=float)
        results[k] = {"values": values, "icer": arr}
    return results


def probabilistic_sensitivity_analysis(df: pd.DataFrame, risk_predictor, n_simulations: int = 5000) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("2. 概率敏感性分析 (PSA)")
    print("=" * 70)
    print(f"  进行 {n_simulations} 次蒙特卡洛模拟...")
    
    np.random.seed(42)
    
    screening_cost = _gamma_from_mean_sd(50, 10, n_simulations)
    intervention_cost = _gamma_from_mean_sd(600, 120, n_simulations)
    management_cost = _gamma_from_mean_sd(1200, 240, n_simulations)
    screening_sensitivity = _beta_from_mean_sd(0.72, 0.08, n_simulations)
    qaly_per_case = _beta_from_mean_sd(0.20, 0.05, n_simulations)
    intervention_effect = _lognormal_from_mean_sd(1.25, 0.15, n_simulations)
    
    rows = []
    for i in range(n_simulations):
        p = {
            "screening_cost": screening_cost[i],
            "intervention_cost": intervention_cost[i],
            "disease_management_cost": management_cost[i],
            "screening_sensitivity": screening_sensitivity[i],
            "qaly_per_case": qaly_per_case[i],
            "intervention_effect": intervention_effect[i],
        }
        
        out = {}
        for s in ["A", "B", "C", "D"]:
            c, e, icer = _calc_strategy_outcome(df, s, p, risk_predictor)
            out[f"cost_{s}"] = c
            out[f"qaly_{s}"] = e
            out[f"icer_{s}"] = icer
        
        for s in ["B", "C", "D"]:
            dc = out[f"cost_{s}"] - out["cost_A"]
            de = out[f"qaly_{s}"] - out["qaly_A"]
            out[f"inc_cost_{s}"] = dc
            out[f"inc_qaly_{s}"] = de
            out[f"inc_icer_{s}"] = dc / de if de > 0 else np.nan
        
        rows.append(out)
    
    psa_df = pd.DataFrame(rows)
    
    print("\n  PSA 关键统计（策略C增量ICER）:")
    c_icer = psa_df["inc_icer_C"].replace([np.inf, -np.inf], np.nan).dropna()
    if not c_icer.empty:
        print(f"    中位数: {c_icer.median():.2f} 元/QALY")
        print(f"    均值: {c_icer.mean():.2f} 元/QALY")
        print(f"    95% CI: [{c_icer.quantile(0.025):.2f}, {c_icer.quantile(0.975):.2f}]")
    else:
        print("    无有效增量ICER数据")
    
    return psa_df


def ceac_from_psa(psa_df: pd.DataFrame, wtp_grid: np.ndarray | None = None) -> pd.DataFrame:
    if wtp_grid is None:
        wtp_grid = np.linspace(1000, 50000, 150)
    
    rows = []
    for wtp in wtp_grid:
        nmb = pd.DataFrame({
            "B": wtp * psa_df["inc_qaly_B"] - psa_df["inc_cost_B"],
            "C": wtp * psa_df["inc_qaly_C"] - psa_df["inc_cost_C"],
            "D": wtp * psa_df["inc_qaly_D"] - psa_df["inc_cost_D"],
        })
        nmb = nmb.replace([np.inf, -np.inf], np.nan).fillna(-1e18)
        best = nmb.idxmax(axis=1)
        rows.append({
            "wtp": float(wtp),
            "prob_B": float((best == "B").mean()),
            "prob_C": float((best == "C").mean()),
            "prob_D": float((best == "D").mean()),
        })
    return pd.DataFrame(rows)


def bootstrap_optimality_test(df: pd.DataFrame, risk_predictor, n_bootstrap: int = 200, wtp: float = 45000.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n" + "=" * 70)
    print("3. Bootstrap 最优策略统计推断")
    print("=" * 70)
    
    np.random.seed(123)
    
    p = {
        "screening_cost": 50.0,
        "intervention_cost": 600.0,
        "disease_management_cost": 1200.0,
        "screening_sensitivity": 0.72,
        "qaly_per_case": 0.20,
        "intervention_effect": 1.25,
    }
    
    out = []
    n = len(df)
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        s = df.iloc[idx].reset_index(drop=True)
        
        cA, eA, _ = _calc_strategy_outcome(s, "A", p, risk_predictor)
        cB, eB, _ = _calc_strategy_outcome(s, "B", p, risk_predictor)
        cC, eC, _ = _calc_strategy_outcome(s, "C", p, risk_predictor)
        cD, eD, _ = _calc_strategy_outcome(s, "D", p, risk_predictor)
        
        nmbB = wtp * (eB - eA) - (cB - cA)
        nmbC = wtp * (eC - eA) - (cC - cA)
        nmbD = wtp * (eD - eA) - (cD - cA)
        
        out.append({
            "delta_nmb_C_minus_B": nmbC - nmbB,
            "delta_nmb_C_minus_D": nmbC - nmbD,
        })
    
    boot_df = pd.DataFrame(out)
    
    summary = []
    for col in ["delta_nmb_C_minus_B", "delta_nmb_C_minus_D"]:
        s = boot_df[col]
        summary.append({
            "comparison": col,
            "mean": s.mean(),
            "median": s.median(),
            "ci_2_5": s.quantile(0.025),
            "ci_97_5": s.quantile(0.975),
            "prob_gt_0": (s > 0).mean(),
        })
    
    summary_df = pd.DataFrame(summary)
    
    print("\n  Bootstrap 结果（C策略相对优势）:")
    for _, r in summary_df.iterrows():
        print(f"    {r['comparison']}: CI=[{r['ci_2_5']:.2f}, {r['ci_97_5']:.2f}], P(>0)={r['prob_gt_0']:.3f}")
    
    return boot_df, summary_df


def visualize_separate_plots(sensitivity_results: dict, psa_df: pd.DataFrame, ceac_df: pd.DataFrame, boot_df: pd.DataFrame, out_base: str) -> None:
    """
    分别输出四张独立图片：
    1. 飓风图 (tornado)
    2. PSA 增量ICER分布直方图 (psa_hist)
    3. CEAC 曲线 (ceac)
    4. Bootstrap 小提琴图 (bootstrap)
    """
    out_dir = Path(out_base).parent
    stem = Path(out_base).stem
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 参数名中文映射（飓风图）
    CN_PARAM = {
        "screening_cost": "筛查成本",
        "intervention_cost": "干预成本",
        "disease_management_cost": "疾病管理成本",
        "screening_sensitivity": "筛查灵敏度",
        "qaly_per_case": "每例QALY增益",
        "intervention_effect": "干预效果",
    }

    # WTP 参考阈值（用于注释标注）
    WTP_THRESHOLD = 15000.0

    # ---------- 1. 飓风图 ----------
    fig1, ax1 = plt.subplots(figsize=(10, 6), facecolor="white")
    baseline = float(sensitivity_results.get("__baseline_icer__", 0.0))
    records = []
    for p_name, vals in sensitivity_results.items():
        if not isinstance(vals, dict):
            continue
        records.append({"param": p_name, "low": np.nanmin(vals["icer"]), "high": np.nanmax(vals["icer"])})
    tor = pd.DataFrame(records)
    if tor.empty:
        tor = pd.DataFrame([{"param": "none", "low": baseline, "high": baseline}])
    tor["range"] = tor["high"] - tor["low"]
    tor = tor.sort_values("range", ascending=True)
    
    y = np.arange(len(tor))
    ax1.barh(y, tor["high"] - baseline, left=baseline, color=COLORS["D"], alpha=0.65, label="高值")
    ax1.barh(y, tor["low"] - baseline, left=baseline, color=COLORS["B"], alpha=0.8, label="低值")
    ax1.axvline(baseline, color="#2c3e50", linestyle="--", linewidth=1.8, zorder=4)
    # 基线标注
    ax1.annotate(f"基线 ICER\n¥{baseline:,.0f}", xy=(baseline, len(tor) - 0.3),
                 xytext=(baseline + abs(baseline) * 0.15, len(tor) - 0.3),
                 fontsize=8.5, color=COLORS["C"], fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=COLORS["C"], lw=0.8), zorder=5)
    # WTP 阈值线标注
    ax1.axvline(WTP_THRESHOLD, color=COLORS["U"], linestyle=":", linewidth=1.5, zorder=3)
    ax1.annotate(f"WTP 阈值 ¥{WTP_THRESHOLD:,.0f}/QALY", xy=(WTP_THRESHOLD, len(tor) - 0.6),
                 xytext=(WTP_THRESHOLD - abs(WTP_THRESHOLD) * 0.2, len(tor) - 0.6),
                 fontsize=8.5, color=COLORS["U"], fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=COLORS["U"], lw=0.8), zorder=5)
    ax1.set_yticks(y)
    tor_cn_labels = [CN_PARAM.get(p, p) for p in tor["param"]]
    ax1.set_yticklabels(tor_cn_labels)
    ax1.set_xlabel("ICER（RMB/QALY）")
    ax1.set_title("单因素敏感性分析", fontweight="bold")
    ax1.grid(True, axis="x", alpha=0.3)
    ax1.xaxis.set_major_formatter(FuncFormatter(_thousands_formatter))
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(out_dir / f"{stem}_tornado.png", dpi=300)
    plt.close(fig1)
    
    # ---------- 2. PSA 增量ICER分布 ----------
    fig2, ax2 = plt.subplots(figsize=(10, 6), facecolor="white")
    s = psa_df["inc_icer_C"].replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        s = pd.Series([0.0])
    # WTP 阈值区域着色
    x_min, x_max = s.quantile(0.001), s.quantile(0.999)
    ax2.axvspan(x_min, WTP_THRESHOLD, alpha=0.12, color="#4b6580", zorder=0, label="低于 WTP 阈值")
    ax2.axvspan(WTP_THRESHOLD, x_max, alpha=0.10, color="#e74c3c", zorder=0, label="高于 WTP 阈值")
    # 直方图
    ax2.hist(s, bins=45, color=COLORS["C"], alpha=0.75, edgecolor="white", density=True, zorder=2)
    # 叠加核密度曲线
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(s.values)
    xs = np.linspace(s.min(), s.max(), 500)
    ax2.plot(xs, kde(xs), color="#2c3e50", linewidth=1.8, zorder=3, label="核密度估计")
    # 统计量标注
    q025, q5, q975 = s.quantile(0.025), s.median(), s.quantile(0.975)
    ax2.axvline(q5, color=COLORS["D"], linewidth=2, zorder=4)
    ax2.axvline(q025, color=COLORS["U"], linestyle="--", linewidth=1.5, zorder=4)
    ax2.axvline(q975, color=COLORS["U"], linestyle="--", linewidth=1.5, zorder=4)
    # 数值标注
    y_top = ax2.get_ylim()[1] * 0.85
    ax2.annotate(f"中位数 = ¥{q5:,.0f}", xy=(q5, y_top), xytext=(q5 + abs(q5)*0.12, y_top),
                 fontsize=9, color=COLORS["D"], fontweight="bold", ha="left",
                 arrowprops=dict(arrowstyle="->", color=COLORS["D"], lw=0.8), zorder=5)
    ax2.annotate(f"95% CI\n[¥{q025:,.0f}, ¥{q975:,.0f}]",
                 xy=(q025, y_top * 0.55), xytext=(q975 + abs(q975)*0.08, y_top * 0.55),
                 fontsize=8.5, color=COLORS["U"], fontweight="bold", ha="left",
                 arrowprops=dict(arrowstyle="->", color=COLORS["U"], lw=0.8), zorder=5)
    ax2.set_title("增量 ICER 的 PSA 分布（策略 C）", fontweight="bold")
    ax2.set_xlabel("增量 ICER（人民币/QALY）")
    ax2.set_ylabel("密度")
    ax2.legend(fontsize=8.5)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.xaxis.set_major_formatter(FuncFormatter(_thousands_formatter))
    fig2.tight_layout()
    fig2.savefig(out_dir / f"{stem}_psa_hist.png", dpi=300)
    plt.close(fig2)
    
    # ---------- 3. CEAC ----------
    fig3, ax3 = plt.subplots(figsize=(10, 6), facecolor="white")
    ax3.plot(ceac_df["wtp"], ceac_df["prob_B"] * 100, label="B：随机40%筛查", color=COLORS["B"], linewidth=2)
    ax3.plot(ceac_df["wtp"], ceac_df["prob_C"] * 100, label="C：前40%精准筛查", color=COLORS["C"], linewidth=2.5)
    ax3.plot(ceac_df["wtp"], ceac_df["prob_D"] * 100, label="D：前40%精准筛查 + 干预", color=COLORS["D"], linewidth=2)
    # 关键 WTP 阈值线
    ax3.axvline(15000, linestyle="--", color=COLORS["U"], linewidth=1.5, alpha=0.7)
    ax3.annotate("¥15,000/QALY", xy=(15000, 98), xytext=(15000, 98),
                 fontsize=8.5, color=COLORS["U"], fontweight="bold", ha="center",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=COLORS["U"], alpha=0.85))
    ax3.axvline(45000, linestyle="--", color=COLORS["U"], linewidth=1.5, alpha=0.7)
    ax3.annotate("¥45,000/QALY", xy=(45000, 98), xytext=(45000, 98),
                 fontsize=8.5, color=COLORS["U"], fontweight="bold", ha="center",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=COLORS["U"], alpha=0.85))
    ax3.set_ylim(0, 105)
    ax3.set_xlabel("支付意愿阈值（人民币/QALY）")
    ax3.set_ylabel("成为最优方案的概率（%）")
    ax3.set_title("成本效果可接受曲线（CEAC）", fontweight="bold")
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3.savefig(out_dir / f"{stem}_ceac.png", dpi=300)
    plt.close(fig3)
    
    # ---------- 4. Bootstrap 小提琴图 ----------
    fig4, ax4 = plt.subplots(figsize=(12, 6), facecolor="white")
    pairs = [("delta_nmb_C_minus_B", "C − B"), ("delta_nmb_C_minus_D", "C − D")]
    rows = []
    for col, label in pairs:
        s_col = boot_df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if s_col.empty:
            s_col = pd.Series([0.0])
        rows.append(
            {
                "label": label,
                "values_million": (s_col / 1e6).values,
                "mean": s_col.mean() / 1e6,
                "median": s_col.median() / 1e6,
                "low": s_col.quantile(0.025) / 1e6,
                "high": s_col.quantile(0.975) / 1e6,
                "p_gt_0": float((s_col > 0).mean()),
                "n": int(len(s_col)),
            }
        )
    
    y_pos = np.arange(len(rows))
    violin_data = [r["values_million"] for r in rows]
    vp = ax4.violinplot(violin_data, positions=y_pos, vert=False, widths=0.7, showmeans=False, showmedians=False, showextrema=False)
    for body in vp["bodies"]:
        body.set_facecolor(COLORS["B"])
        body.set_edgecolor(COLORS["C"])
        body.set_alpha(0.28)
    
    for i, r in enumerate(rows):
        ax4.hlines(i, r["low"], r["high"], color=COLORS["C"], linewidth=2.2)
        ax4.scatter(r["mean"], i, color=COLORS["D"], s=42, zorder=3, label="均值" if i == 0 else None)
        ax4.scatter(r["median"], i, color=COLORS["C"], s=34, marker="D", zorder=3, label="中位数" if i == 0 else None)
        # 在图上直接标注 P(>0) 值
        ax4.text(r["high"] + 0.03, i,
                 f"P(ΔNMB>0)={r['p_gt_0']:.3f}",
                 va="center", fontsize=9.5, color=COLORS["D"], fontweight="bold")
        # 下方标注 95% CI
        ax4.text(r["high"] + 0.03, i - 0.22,
                 f"95% CI [{r['low']:.2f}, {r['high']:.2f}]",
                 va="center", fontsize=8, color=COLORS["U"])
    
    # 加粗零线
    ax4.axvline(0, color="#2c3e50", linestyle="--", linewidth=1.8, zorder=4, label="零增益线")
    # 在零线附近添加标注
    y_center = len(rows) - 0.5
    ax4.annotate("← C 劣势区  |  C 优势区 →", xy=(0, -0.3),
                 xytext=(0, -0.3), fontsize=9, color=COLORS["C"],
                 ha="center", va="top",
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=COLORS["B"], alpha=0.8))
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels([r["label"] for r in rows])
    ax4.set_xlabel("WTP=4.5 万时的 ΔNMB（百万元）")
    ax4.set_title("策略优越性的自助法推断", fontweight="bold")
    ax4.grid(True, axis="x", alpha=0.3)
    ax4.xaxis.set_major_formatter(FuncFormatter(_thousands_formatter))
    
    lows = np.array([r["low"] for r in rows])
    highs = np.array([r["high"] for r in rows])
    means = np.array([r["mean"] for r in rows])
    xmin_val = min(lows.min(), means.min())
    xmax_val = max(highs.max(), means.max())
    pad = (xmax_val - xmin_val) * 0.22 if xmax_val > xmin_val else 1.0
    ax4.set_xlim(xmin_val - pad, xmax_val + pad)
    ax4.set_ylim(-0.8, len(rows) - 0.2)
    ax4.legend(loc="lower right", fontsize=8.8, frameon=True)
    
    fig4.tight_layout()
    fig4.savefig(out_dir / f"{stem}_bootstrap.png", dpi=300)
    plt.close(fig4)


def main() -> None:
    print("\n" + "=" * 70)
    print("高级经济学分析：PSA + CEAC + Bootstrap")
    print("=" * 70)
    
    data_path = PROJECT_ROOT / "01 cleaned" / "charls_latest_cross_section_2018.csv"
    if not data_path.exists():
        print(f"错误：数据文件不存在 {data_path}")
        return
    
    df = pd.read_csv(data_path)
    
    # 构造目标变量（统一到十四病种口径）
    df["target"] = _build_any14_target(df)
    
    # 确保后续预测时需要的特征列存在
    required_features = ['age', 'bmi', 'comorbidity_count', 'gender', 'smoking_status', 'drinking']
    for col in required_features:
        if col not in df.columns:
            if col in ['age', 'bmi']:
                df[col] = np.nan
            elif col == 'comorbidity_count':
                df[col] = 0
            else:
                df[col] = 'unknown'
    
    df = df.dropna(subset=["target"]).reset_index(drop=True)
    print(f"有效样本量: {len(df)}")
    
    risk_predictor = _get_risk_predictor(df)
    
    sensitivity_results = one_way_sensitivity_analysis(df, risk_predictor)
    psa_df = probabilistic_sensitivity_analysis(df, risk_predictor, n_simulations=5000)
    ceac_df = ceac_from_psa(psa_df)
    boot_df, boot_summary = bootstrap_optimality_test(df, risk_predictor, n_bootstrap=200, wtp=45000)
    
    # 保存结果 CSV
    psa_out = PROJECT_ROOT / "01 cleaned" / "psa_simulation_results.csv"
    ceac_out = PROJECT_ROOT / "01 cleaned" / "ceac_curve_results.csv"
    boot_out = PROJECT_ROOT / "01 cleaned" / "bootstrap_optimality_results.csv"
    boot_summary_out = PROJECT_ROOT / "01 cleaned" / "bootstrap_optimality_summary.csv"
    
    psa_df.to_csv(psa_out, index=False)
    ceac_df.to_csv(ceac_out, index=False)
    boot_df.to_csv(boot_out, index=False)
    boot_summary.to_csv(boot_summary_out, index=False)
    
    print(f"✓ PSA结果已保存: {psa_out}")
    print(f"✓ CEAC结果已保存: {ceac_out}")
    print(f"✓ Bootstrap结果已保存: {boot_out}")
    print(f"✓ Bootstrap汇总已保存: {boot_summary_out}")
    
    # 输出四张独立图片
    out_base = PROJECT_ROOT / "03 visualizations" / "sensitivity_and_economic_analysis.png"
    visualize_separate_plots(sensitivity_results, psa_df, ceac_df, boot_df, str(out_base))
    print(f"✓ 四张独立图已保存至: {out_base.parent}")
    
    print("\n完成：新增统计推断内容已同步到代码与可视化。")


if __name__ == "__main__":
    main()
