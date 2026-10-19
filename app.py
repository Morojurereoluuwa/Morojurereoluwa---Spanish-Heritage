"""
Hispanic Heritage Historical Data Explorer
Self-contained Streamlit app: historical World Bank data + student function lab.
"""
from __future__ import annotations

import io
import math
import re
import textwrap
from datetime import date, datetime
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import sympy as sp
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Hispanic Heritage Historical Data Explorer", page_icon="📊", layout="wide")

COUNTRIES = {
    "Mexico (MX)": "MEX", "Brazil (BR)": "BRA", "Argentina (AR)": "ARG",
    "Colombia (CO)": "COL", "Chile (CL)": "CHL", "Peru (PE)": "PER",
    "Venezuela (VE)": "VEN", "Ecuador (EC)": "ECU", "Uruguay (UY)": "URY",
}
INDICATORS = {
    "Life expectancy at birth": ("SP.DYN.LE00.IN", "years", lambda x: x >= 0,
        "Average number of years a newborn would live if current age-specific mortality rates persisted."),
    "Unemployment, total (national estimate)": ("SL.UEM.TOTL.NE.ZS", "% of labor force", lambda x: 0 <= x <= 100,
        "Share of the labor force without work but available for and seeking employment, using national estimates."),
    "GDP per capita (current US$)": ("NY.GDP.PCAP.CD", "current US dollars per person", lambda x: x >= 0,
        "Gross domestic product divided by midyear population. It is not average wealth or household income."),
    "Birth rate, crude": ("SP.DYN.CBRT.IN", "births per 1,000 people per year", lambda x: x >= 0,
        "Annual number of live births per 1,000 people; this is a rate, not a birth count."),
    "Net migration": ("SM.POP.NETM", "people (net estimate)", lambda x: True,
        "Difference between the number of immigrants and emigrants over a period, including statistical adjustments; not an emigration count."),
    "Intentional homicide rate": ("VC.IHR.PSRC.P5", "intentional homicides per 100,000 people", lambda x: x >= 0,
        "Intentional homicide victims per 100,000 population."),
    "Consumer price inflation": ("FP.CPI.TOTL.ZG", "% annual change", lambda x: True,
        "Annual percentage change in the consumer price index."),
    "Central government debt": ("GC.DOD.TOTL.GD.ZS", "% of GDP", lambda x: True,
        "Central government debt as a percentage of GDP; coverage and comparability may vary."),
    "Total reserves including gold": ("FI.RES.TOTL.CD", "current US dollars", lambda x: x >= 0,
        "Total reserves including gold, valued in current US dollars."),
    "External balance on goods and services": ("NE.RSB.GNFS.ZS", "% of GDP", lambda x: True,
        "External balance on goods and services as a percentage of GDP; it is not the full current account."),
}
WB_API = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?date={start}:{end}&format=json&per_page=20000"
WB_PAGE = "https://data.worldbank.org/indicator/{indicator}?locations={country}"
CURRENT_YEAR = date.today().year
START_YEAR, END_YEAR = CURRENT_YEAR - 70, CURRENT_YEAR - 1

CONTEXTS = {
    "Business": ("Revenue or demand", "time", "years", "value", "dollars / units", "A business measure evolving over time."),
    "Engineering": ("Position / system output", "time", "seconds", "output", "meters / units", "A physical system or engineered device."),
    "Medical": ("Drug concentration", "time", "hours", "concentration", "mg/L", "A concentration or patient-monitoring measure; not medical advice."),
    "Sports science": ("Athlete speed", "time", "seconds", "speed", "m/s", "A movement or performance measure."),
    "Biology": ("Population / cell count", "time", "days", "population", "cells / organisms", "A population or cell-culture measure."),
    "Bioengineering": ("Prosthetic gait signal", "gait phase", "% cycle", "signal", "normalized units", "A simplified signal associated with a prosthetic or assistive device."),
    "Economic forecasting": ("Indicator level", "time", "years", "indicator", "indicator units", "A mathematical scenario, not a causal economic forecast."),
    "Weather": ("Temperature", "time", "hours", "temperature", "°C", "A simplified weather variable."),
    "Custom": ("Custom function", "x", "x-units", "f(x)", "y-units", "Define your own interpretation and units."),
}
SUGGESTED = {
    "Business": {"Polynomial": "0.04*x**2 + 2*x + 10", "Exponential": "10*exp(0.08*x)", "Logarithmic": "5*log(x+1)+10"},
    "Engineering": {"Polynomial": "-4.9*x**2 + 20*x + 1", "Trigonometric": "2*sin(x)+3", "Rational": "10/(x+1)"},
    "Medical": {"Exponential": "8*exp(-0.25*x)", "Polynomial": "0.2*x**2 - 2*x + 8", "Rational": "10/(x+1)"},
    "Sports science": {"Polynomial": "-0.5*x**2 + 4*x", "Trigonometric": "3*sin(x)+5", "Exponential": "2*exp(0.1*x)"},
    "Biology": {"Exponential": "2*exp(0.15*x)", "Logarithmic": "4*log(x+1)+2", "Polynomial": "0.02*x**2+0.3*x+2"},
    "Bioengineering": {"Trigonometric": "1.5*sin(2*pi*x)+2", "Polynomial": "0.4*x**2-1.2*x+2", "Rational": "1/(x+1)+1"},
    "Economic forecasting": {"Polynomial": "0.03*x**2 + 0.5*x + 10", "Exponential": "10*exp(0.03*x)", "Logarithmic": "2*log(x+1)+10"},
    "Weather": {"Trigonometric": "10*sin(2*pi*x/24)+15", "Polynomial": "-0.03*x**2+0.5*x+10", "Exponential": "15*exp(-0.02*x)"},
    "Custom": {"Polynomial": "x**3 - 2*x", "Rational": "(x+1)/(x-2)", "Trigonometric": "sin(x)+x"},
}
x_symbol = sp.Symbol("x", real=True)

def wb_url(country: str, indicator: str) -> str:
    return WB_PAGE.format(indicator=indicator, country=country)

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_world_bank(country: str, indicator: str, start: int, end: int) -> tuple[pd.DataFrame, str]:
    """Retrieve official World Bank observations; errors are returned as explanatory text."""
    url = WB_API.format(country=country, indicator=indicator, start=start, end=end)
    try:
        response = requests.get(url, timeout=(5, 25), headers={"User-Agent": "HispanicHeritageDataExplorer/1.0"})
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
            message = payload[0].get("message", "Unexpected or empty API response") if isinstance(payload, list) and payload and isinstance(payload[0], dict) else "Unexpected or empty API response"
            return pd.DataFrame(columns=["year", "value"]), str(message)
        rows = [{"year": int(r["date"]), "value": r.get("value")} for r in payload[1] if r.get("date")]
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["year", "value"]), "The API returned no observations for this request."
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"]).sort_values("year").reset_index(drop=True)
        return df, f"Retrieved from World Bank API at {datetime.now().strftime('%Y-%m-%d %H:%M')} local time."
    except requests.Timeout:
        return pd.DataFrame(columns=["year", "value"]), "Request timed out. Retry later or check network access."
    except requests.RequestException as exc:
        return pd.DataFrame(columns=["year", "value"]), f"Network/API error: {exc}"
    except (ValueError, TypeError, KeyError) as exc:
        return pd.DataFrame(columns=["year", "value"]), f"Could not parse API response: {exc}"

def clean_numeric_frame(df: pd.DataFrame, validator) -> tuple[pd.DataFrame, list[str]]:
    issues = []
    if not {"year", "value"}.issubset(df.columns):
        return pd.DataFrame(columns=["year", "value"]), ["Table must contain year and value columns."]
    years = pd.to_numeric(df["year"], errors="coerce")
    vals = pd.to_numeric(df["value"], errors="coerce")
    keep = years.notna() & vals.notna()
    if (~keep).any():
        issues.append(f"Dropped {(~keep).sum()} row(s) with nonnumeric or empty year/value.")
    out = pd.DataFrame({"year": years[keep], "value": vals[keep]})
    nonint = (out["year"] % 1 != 0)
    if nonint.any():
        issues.append(f"Dropped {int(nonint.sum())} non-integer year row(s).")
        out = out.loc[~nonint].copy()
    out["year"] = out["year"].astype(int)
    duplicates = out["year"].duplicated(keep=False)
    if duplicates.any():
        issues.append("Duplicate years found; keep one value per year before fitting.")
        out = out.loc[~duplicates].copy()
    invalid = ~out["value"].map(lambda v: bool(validator(float(v))))
    if invalid.any():
        issues.append(f"Dropped {int(invalid.sum())} value(s) outside the indicator's basic valid range.")
        out = out.loc[~invalid].copy()
    out = out.sort_values("year").reset_index(drop=True)
    return out, issues

def fit_polynomial(df: pd.DataFrame, degree: int):
    clean = df.sort_values("year")
    if clean["year"].nunique() < degree + 1:
        raise ValueError(f"Degree {degree} requires at least {degree+1} distinct valid years; found {clean['year'].nunique()}.")
    base = int(clean["year"].min())
    t = clean["year"].to_numpy(dtype=float) - base
    y = clean["value"].to_numpy(dtype=float)
    coef = np.polynomial.polynomial.polyfit(t, y, degree)
    pred = np.polynomial.polynomial.polyval(t, coef)
    r2 = float(r2_score(y, pred)) if len(y) > 1 else float("nan")
    mae = float(mean_absolute_error(y, pred))
    residuals = y - pred
    return base, coef, pred, r2, mae, residuals

def polynomial_text(coef, base, degree):
    terms = []
    for power, c in enumerate(coef):
        if abs(c) < 1e-12:
            continue
        if power == 0:
            term = f"{c:.8g}"
        elif power == 1:
            term = f"{c:+.8g}(year − {base})"
        else:
            term = f"{c:+.8g}(year − {base})^{power}"
        terms.append(term)
    return "f(year) = " + " ".join(terms).replace("+ -", "− ")

def model_value(year: float, base: int, coef) -> float:
    # The centered time transformation is applied exactly once.
    return float(np.polynomial.polynomial.polyval(float(year) - base, coef))

def parse_custom_years(text: str) -> tuple[list[int], list[str]]:
    years, errors = [], []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            val = float(part)
            if not val.is_integer():
                errors.append(f"{part!r} is not an integer year.")
            elif not 1950 <= int(val) <= 2100:
                errors.append(f"{part!r} is outside 1950–2100.")
            else:
                years.append(int(val))
        except ValueError:
            errors.append(f"{part!r} is not a valid year.")
    return sorted(set(years)), errors

def parse_expr(text: str):
    """Parse familiar x notation while allowing only mathematical symbols/functions."""
    allowed = {"x": x_symbol, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp,
               "log": sp.log, "sqrt": sp.sqrt, "pi": sp.pi, "E": sp.E}
    try:
        expr = sp.sympify(text, locals=allowed, evaluate=True)
        if expr.free_symbols - {x_symbol}:
            raise ValueError("Use x as the only variable.")
        if expr.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
            raise ValueError("Expression contains an undefined or infinite constant.")
        return expr
    except Exception as exc:
        raise ValueError(f"Could not parse function. Use notation such as x**2, sin(x), exp(x), log(x), or (x+1)/(x-2). Details: {exc}")

def numeric_fn(expr):
    return sp.lambdify(x_symbol, expr, modules=["numpy"])

def safe_eval(fn, x):
    with np.errstate(all="ignore"):
        result = np.asarray(fn(np.asarray(x, dtype=float)))
    if result.ndim == 0:
        return float(result)
    return result.astype(float)

def approximate_features(expr, lo, hi, n=1800):
    f = numeric_fn(expr)
    d1, d2 = sp.diff(expr, x_symbol), sp.diff(expr, x_symbol, 2)
    f1, f2 = numeric_fn(d1), numeric_fn(d2)
    grid = np.linspace(lo, hi, n)
    y, dy, ddy = safe_eval(f, grid), safe_eval(f1, grid), safe_eval(f2, grid)
    if np.ndim(y) == 0: y = np.full_like(grid, y)
    if np.ndim(dy) == 0: dy = np.full_like(grid, dy)
    if np.ndim(ddy) == 0: ddy = np.full_like(grid, ddy)
    y = np.where(np.isfinite(y) & (np.abs(y) < 1e12), y, np.nan)
    dy = np.where(np.isfinite(dy) & (np.abs(dy) < 1e12), dy, np.nan)
    ddy = np.where(np.isfinite(ddy) & (np.abs(ddy) < 1e12), ddy, np.nan)
    def zeros_of(arr):
        found = []
        for i in range(len(grid)-1):
            if np.isfinite(arr[i]) and np.isfinite(arr[i+1]) and arr[i] * arr[i+1] < 0:
                found.append(float((grid[i]+grid[i+1])/2))
            elif np.isfinite(arr[i]) and abs(arr[i]) < 1e-5:
                found.append(float(grid[i]))
        return dedupe_close(found, (hi-lo)/n*2)
    critical = zeros_of(dy)
    inflect_candidates = zeros_of(ddy)
    extrema, inflections = [], []
    for p in critical:
        eps = max((hi-lo)/10000, 1e-5)
        try:
            left, right = float(safe_eval(f1, p-eps)), float(safe_eval(f1, p+eps))
            typ = "local maximum" if left > 0 and right < 0 else "local minimum" if left < 0 and right > 0 else "stationary point (no sign change)"
            if typ != "stationary point (no sign change)":
                extrema.append((p, float(safe_eval(f, p)), typ))
        except Exception: pass
    for p in inflect_candidates:
        eps = max((hi-lo)/10000, 1e-5)
        try:
            left, right = float(safe_eval(f2, p-eps)), float(safe_eval(f2, p+eps))
            if np.isfinite(left) and np.isfinite(right) and left*right < 0:
                inflections.append((p, float(safe_eval(f, p))))
        except Exception: pass
    return grid, y, dy, ddy, critical, extrema, inflections

def dedupe_close(values, tol):
    result = []
    for v in sorted(values):
        if not result or abs(v-result[-1]) > tol:
            result.append(v)
    return result

def derivative_text(expr):
    return sp.sstr(sp.simplify(expr)).replace("**", "^")

def make_chart(df, fitted, base, coef, degree, sample_step, single_result, batch_results,
               x_label, y_label, chart_title, comparison=None):
    fig, ax = plt.subplots(figsize=(10, 5.4))
    if not df.empty:
        ax.scatter(df["year"], df["value"], label="Observed source / edited data", s=24, zorder=3)
        years = df["year"].to_numpy()
        lo, hi = int(years.min()), int(years.max())
        dense = np.linspace(lo, hi, max(250, (hi-lo)*4+1))
        ax.plot(dense, [model_value(y, base, coef) for y in dense], label=f"Polynomial fit (degree {degree})", linewidth=2)
        sample = np.arange(lo, hi+1, sample_step)
        ax.scatter(sample, [model_value(y, base, coef) for y in sample], marker="x", s=34, label=f"Model samples every {sample_step} year(s)")
        ax.axvline(hi, linestyle="--", linewidth=1, label="Historical boundary")
        ext = np.linspace(hi, max(hi+1, min(hi+20, 2100)), 100)
        if len(ext) > 1:
            ax.plot(ext, [model_value(y, base, coef) for y in ext], linestyle=":", label="Extrapolation (illustrative)")
    if comparison is not None and not comparison.empty:
        ax.plot(comparison["year"], comparison["value"], linestyle="--", marker=".", label="Comparison country observations")
    if single_result is not None:
        ax.scatter([single_result["year"]], [single_result["estimate"]], marker="D", s=70, label="Single-year estimate")
    if batch_results is not None and not batch_results.empty:
        ax.scatter(batch_results["year"], batch_results["estimate"], marker="^", s=50, label="Batch estimates")
    ax.set(title=chart_title, xlabel=x_label, ylabel=y_label)
    ax.grid(True, alpha=.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig

def make_pdf(summary: dict, chart_bytes: bytes, preview: pd.DataFrame, results: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(letter), rightMargin=28, leftMargin=28, topMargin=24, bottomMargin=24)
    styles = getSampleStyleSheet()
    story = [Paragraph("Hispanic Heritage Historical Data Explorer", styles["Title"]), Spacer(1, 8)]
    for key, value in summary.items():
        story.append(Paragraph(f"<b>{key}:</b> {str(value)}", styles["BodyText"]))
    story.append(Spacer(1, 10))
    story.append(RLImage(io.BytesIO(chart_bytes), width=650, height=340))
    story.append(Spacer(1, 8))
    def add_table(df, title):
        story.append(Paragraph(title, styles["Heading2"]))
        if df is None or df.empty:
            story.append(Paragraph("No rows available.", styles["BodyText"]))
            return
        show = df.head(18).copy()
        data = [list(map(str, show.columns))] + [[str(v)[:32] for v in row] for row in show.astype(str).values.tolist()]
        tbl = Table(data, repeatRows=1)
        tbl.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#f2d35e")),
                                 ("GRID",(0,0),(-1,-1),.35,colors.grey),("FONTSIZE",(0,0),(-1,-1),7),
                                 ("VALIGN",(0,0),(-1,-1),"TOP")]))
        story.append(tbl)
    add_table(preview, "Data preview (up to 18 rows)")
    add_table(results, "Calculated-year results")
    doc.build(story)
    return buffer.getvalue()

st.title("📊 Hispanic Heritage Historical Data Explorer")
st.caption("HISPANIC HERITAGE DATA LAB • World Bank historical data + interactive mathematics/function lab")
st.markdown("Brazil is included as a **Latin American comparison**; the country list is not limited to Spanish-speaking countries.")

with st.sidebar:
    st.header("Historical data")
    country_label = st.selectbox("Featured country", list(COUNTRIES))
    country_code = COUNTRIES[country_label]
    indicator_label = st.selectbox("Indicator", list(INDICATORS))
    indicator_code, units, validator, definition = INDICATORS[indicator_label]
    degree = st.slider("Polynomial degree", 3, 8, 3)
    sample_step = st.slider("Graph sample increment (years)", 1, 10, 5)
    horizon = st.slider("Forward display horizon (years)", 0, 50, 10)
    comparison_label = st.selectbox("Comparison country (same indicator)", ["None"] + [c for c in COUNTRIES if c != country_label])
    printer_mode = st.checkbox("Printer-friendly report view", value=False)

st.subheader("1. Retrieve official historical data")
st.write(f"Requested range: **{START_YEAR}–{END_YEAR}** inclusive (current year minus 70 through current year minus 1).")
source_df, fetch_message = fetch_world_bank(country_code, indicator_code, START_YEAR, END_YEAR)
st.caption(f"Source: World Bank API • Retrieved/requested on {date.today().isoformat()} • Indicator code: {indicator_code}")
st.markdown(f"**Definition:** {definition}  \n**Units:** {units}  \n**Source link:** [{indicator_label}]({wb_url(country_code, indicator_code)})")
if fetch_message and source_df.empty:
    st.error(fetch_message)
elif fetch_message:
    st.info(fetch_message)
source_valid, source_issues = clean_numeric_frame(source_df, validator)
if source_issues:
    st.warning("Source-data validation: " + " ".join(source_issues))
if not source_valid.empty:
    st.success(f"Actual coverage: {source_valid.year.min()}–{source_valid.year.max()} • {len(source_valid)} valid observations (not guaranteed to be 70).")
else:
    st.warning("No valid observations are currently available. Check the API or choose another indicator/country.")

# Keep source data and editable scenario data separate and reset on selector changes.
data_key = f"{country_code}|{indicator_code}"
if st.session_state.get("_data_key") != data_key:
    st.session_state["_data_key"] = data_key
    st.session_state["working_data"] = source_valid.copy()
    st.session_state["data_edited"] = False
    st.session_state.pop("single_result", None)
    st.session_state.pop("batch_result", None)
if "working_data" not in st.session_state:
    st.session_state["working_data"] = source_valid.copy()
working = st.session_state["working_data"].copy()

st.subheader("2. Validate and edit the working data")
st.write("Edit the year/value table below. Source data stays separate. Duplicate years are rejected; missing/nonnumeric rows and invalid indicator values are excluded from model fitting.")
edited = st.data_editor(working, num_rows="dynamic", use_container_width=True, key=f"editor_{data_key}")
col_reset, col_status = st.columns([1, 3])
with col_reset:
    if st.button("Reset to source", use_container_width=True):
        st.session_state["working_data"] = source_valid.copy()
        st.session_state["data_edited"] = False
        st.session_state.pop("single_result", None)
        st.session_state.pop("batch_result", None)
        st.rerun()
with col_status:
    changed = not edited.reset_index(drop=True).equals(source_valid.reset_index(drop=True))
    st.session_state["data_edited"] = bool(changed)
    if changed:
        st.warning("USER-MODIFIED SCENARIO — calculations and exports use the edited table.")
    else:
        st.caption("Working table currently matches the retrieved source table.")
validated, edit_issues = clean_numeric_frame(edited, validator)
if edit_issues:
    st.warning("Working-data validation: " + " ".join(edit_issues))
if len(validated) < degree + 1:
    st.error(f"Cannot fit degree {degree}: need at least {degree+1} distinct valid years; currently {len(validated)}.")
fit_ok = False
if len(validated) >= degree + 1:
    try:
        base, coef, fitted, r2, mae, residuals = fit_polynomial(validated, degree)
        fit_ok = True
        equation = polynomial_text(coef, base, degree)
        st.subheader("3. Polynomial fit and in-sample diagnostics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Valid observations", len(validated))
        m2.metric("R² (in-sample)", f"{r2:.5f}")
        m3.metric("MAE (in-sample)", f"{mae:.5g}")
        m4.metric("Mean observed value", f"{validated.value.mean():.6g}")
        st.code(equation)
        st.caption(f"Centered time: t = year − {base}. Polynomial degree: {degree}. Historical range: {validated.year.min()}–{validated.year.max()}. Metrics describe the same data used to fit the model; they are not out-of-sample guarantees.")
        residual_table = validated[["year"]].copy()
        residual_table["observed"] = validated.value.to_numpy()
        residual_table["fitted"] = fitted
        residual_table["residual (observed − fitted)"] = residuals
        with st.expander("Residuals and model details"):
            st.dataframe(residual_table, use_container_width=True)
            st.write("Uncertainty note: residuals and MAE describe typical in-sample error. Polynomial extrapolation can become implausible quickly; no validated 95% prediction interval is claimed.")
        # Chronological holdout: reserve final ~20% only when enough points remain for a meaningful fit.
        holdout_n = max(1, int(round(len(validated)*0.2)))
        train = validated.iloc[:-holdout_n]
        test = validated.iloc[-holdout_n:]
        if len(train) >= degree+1 and len(test) >= 1:
            hb, hc, _, _, _, _ = fit_polynomial(train, degree)
            hp = np.array([model_value(y, hb, hc) for y in test.year])
            st.write(f"**Chronological holdout MAE:** {mean_absolute_error(test.value, hp):.6g} ({len(test)} later observation(s); separate from in-sample metrics).")
        else:
            st.caption("Chronological holdout score unavailable: too few earlier observations remain to fit this degree.")
    except (ValueError, np.linalg.LinAlgError) as exc:
        st.error(str(exc))
        fit_ok = False

# Comparison data for the same indicator.
comparison_df = pd.DataFrame(columns=["year", "value"])
if comparison_label != "None":
    comp_code = COUNTRIES[comparison_label]
    raw_comp, comp_message = fetch_world_bank(comp_code, indicator_code, START_YEAR, END_YEAR)
    comparison_df, _ = clean_numeric_frame(raw_comp, validator)
    with st.expander("Comparison-country coverage"):
        st.write(f"{comparison_label}: {len(comparison_df)} observations.")
        st.caption("Country series may have different missing years, coverage, and statistical definitions. Compare only the same indicator and do not interpret gaps as zero.")
        if comp_message and raw_comp.empty: st.warning(comp_message)

single_result = st.session_state.get("single_result")
batch_result = st.session_state.get("batch_result")
if fit_ok:
    st.subheader("4. Calculate fitted values")
    c1, c2 = st.columns(2)
    with c1:
        one_year = st.number_input("Single year (1950–2100)", min_value=1950, max_value=2100, value=int(validated.year.max()), step=1)
        if st.button("Calculate Value"):
            val = model_value(one_year, base, coef)
            st.session_state["single_result"] = {"year": int(one_year), "estimate": val,
                "classification": "within fitted historical range" if validated.year.min() <= one_year <= validated.year.max() else "outside fitted historical range"}
            single_result = st.session_state["single_result"]
    with c2:
        batch_mode = st.radio("Batch input", ["Year range", "Custom years"], horizontal=True)
        if batch_mode == "Year range":
            range_col1, range_col2, range_col3 = st.columns(3)
            with range_col1: range_start = st.number_input("Start", 1950, 2100, int(validated.year.max()), key="batch_start")
            with range_col2: range_end = st.number_input("End", 1950, 2100, min(2100, int(validated.year.max())+10), key="batch_end")
            with range_col3: range_step = st.selectbox("Step", list(range(1,11)), index=0)
            custom_text = ""
        else:
            custom_text = st.text_input("Comma-separated years", value="2000, 2010, 2020")
            range_start, range_end, range_step = 0, -1, 1
        if st.button("Calculate Batch"):
            errors = []
            if batch_mode == "Year range":
                if range_start > range_end: errors.append("Start year must be less than or equal to end year.")
                years = list(range(int(range_start), int(range_end)+1, int(range_step))) if not errors else []
            else:
                years, errors = parse_custom_years(custom_text)
            if errors:
                st.error(" ".join(errors))
            elif not years:
                st.error("Enter at least one valid year.")
            else:
                rows = []
                for yr in years:
                    value = model_value(yr, base, coef)
                    rows.append({"year": yr, "estimate": value,
                                 "units": units,
                                 "classification": "within fitted historical range" if validated.year.min() <= yr <= validated.year.max() else "outside fitted historical range"})
                st.session_state["batch_result"] = pd.DataFrame(rows)
                batch_result = st.session_state["batch_result"]
    # invalidate results if the fitting basis changed
    signature = (country_code, indicator_code, degree, tuple(validated.year.tolist()), tuple(np.round(validated.value.to_numpy(), 12)))
    if st.session_state.get("_fit_signature") != signature:
        st.session_state["_fit_signature"] = signature
        st.session_state.pop("single_result", None)
        st.session_state.pop("batch_result", None)
        single_result = None
        batch_result = None
    if single_result:
        st.write(f"**Single-year estimate:** {single_result['year']} = {single_result['estimate']:.8g} {units}; {single_result['classification']}. This is a fitted value, not an actual observation.")
    if batch_result is not None and not batch_result.empty:
        st.dataframe(batch_result, use_container_width=True)
    if not validated.empty:
        plot_end = min(2100, int(validated.year.max()) + horizon)
        fig = make_chart(validated, fitted, base, coef, degree, sample_step, single_result, batch_result,
                         "Year", units, f"{indicator_label} — {country_label}", comparison_df)
        # Extend the displayed extrapolation to the selected horizon.
        if horizon > 0:
            dense_ext = np.linspace(int(validated.year.max()), plot_end, max(30, horizon*4))
            fig.axes[0].plot(dense_ext, [model_value(y, base, coef) for y in dense_ext], linestyle=":", label=f"Display horizon ({horizon} years)")
            fig.axes[0].legend(fontsize=8)
        st.pyplot(fig, use_container_width=True)
        chart_buffer = io.BytesIO()
        fig.savefig(chart_buffer, format="png", dpi=150, bbox_inches="tight")
        chart_png = chart_buffer.getvalue()
        plt.close(fig)
        extreme_year = plot_end
        if horizon > 0:
            extreme_val = model_value(extreme_year, base, coef)
            scale = max(float(np.nanmax(np.abs(validated.value))), 1.0)
            if not np.isfinite(extreme_val) or abs(extreme_val) > 100*scale:
                st.warning(f"Potentially implausible polynomial extrapolation at {extreme_year}: {extreme_val:.5g} {units}. It is not clipped; interpret cautiously.")
    else:
        chart_png = b""
else:
    chart_png = b""

st.subheader("5. Mathematics pathway and function lab")
course = st.selectbox("Assigned mathematics pathway", ["Honors Algebra 2", "Precalculus", "Calculus"], index=0)
context = st.selectbox("Context", list(CONTEXTS))
suggested_name = CONTEXTS[context][0]
independent_default, xunit_default = CONTEXTS[context][1], CONTEXTS[context][2]
dependent_default, yunit_default = CONTEXTS[context][3], CONTEXTS[context][4]
context_description_default = CONTEXTS[context][5]
description = st.text_input("Context description", value=context_description_default, key=f"desc_{context}")
vcol1, vcol2, vcol3, vcol4 = st.columns(4)
with vcol1: x_name = st.text_input("Independent variable", independent_default, key=f"xn_{context}")
with vcol2: x_units = st.text_input("Independent units", xunit_default, key=f"xu_{context}")
with vcol3: y_name = st.text_input("Dependent variable", dependent_default, key=f"yn_{context}")
with vcol4: y_units = st.text_input("Dependent units", yunit_default, key=f"yu_{context}")
function_source = st.radio("Function source", ["Suggested function", "Build from parameters", "Enter my own function", "Use historical curve of best fit"], horizontal=True)
if function_source == "Suggested function":
    family = st.selectbox("Suggested function family", list(SUGGESTED[context]))
    function_text = SUGGESTED[context][family]
elif function_source == "Build from parameters":
    family = st.selectbox("Parameter model family", ["Polynomial (quadratic)", "Rational", "Exponential", "Logarithmic", "Trigonometric"])
    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1: a = st.number_input("Parameter a", value=1.0, key=f"a_{context}_{family}")
    with pcol2: b = st.number_input("Parameter b", value=1.0, key=f"b_{context}_{family}")
    with pcol3: c = st.number_input("Parameter c", value=0.0, key=f"c_{context}_{family}")
    if family == "Polynomial (quadratic)": function_text = f"{a}*x**2 + {b}*x + {c}"
    elif family == "Rational": function_text = f"{a}/(x - ({b})) + {c}"
    elif family == "Exponential": function_text = f"{a}*exp({b}*x) + {c}"
    elif family == "Logarithmic": function_text = f"{a}*log(x - ({b})) + {c}"
    else: function_text = f"{a}*sin({b}*x) + {c}"
elif function_source == "Enter my own function":
    function_text = st.text_input("Function f(x) (use ** for powers)", value="x**2 - 4*x + 3")
else:
    if fit_ok:
        # Model transformation: x is the calendar year and the centering is applied only here.
        function_text = " + ".join([f"({float(c):.12g})*(x-({base}))**{i}" for i,c in enumerate(coef)])
        st.caption(f"Historical fit uses x = calendar year; centered base year is {base}.")
    else:
        function_text = "x**2"
        st.warning("Historical fit unavailable; using a placeholder until enough valid data are available.")
st.code(f"f(x) = {function_text}")
try:
    expr = parse_expr(function_text)
    d1_expr, d2_expr = sp.simplify(sp.diff(expr, x_symbol)), sp.simplify(sp.diff(expr, x_symbol, 2))
    st.markdown(f"**Function:** \\(f(x) = {sp.latex(expr)}\\)")
    st.markdown(f"**First derivative:** \\(f'(x) = {sp.latex(d1_expr)}\\)")
    st.markdown(f"**Second derivative:** \\(f''(x) = {sp.latex(d2_expr)}\\)")
    if course == "Honors Algebra 2":
        st.info("Algebra 2 focus: identify domain/range from the chosen interval, graph trends, end behavior where meaningful, and average rate of change. Derivatives are shown as optional preview.")
    elif course == "Precalculus":
        st.info("Precalculus focus: compare degree/coefficient effects, approximate turning points, and use changing average rates to discuss concavity.")
    else:
        st.info("Calculus focus: derivatives, tangent/secant lines, critical points, derivative tests, extrema, concavity, and numerically verified inflection points.")
    default_lo, default_hi = (0.0, 10.0)
    if function_source == "Use historical curve of best fit" and fit_ok:
        default_lo, default_hi = float(validated.year.min()), float(validated.year.max())
    lo_col, hi_col = st.columns(2)
    with lo_col: x_lo = st.number_input("Analysis interval start", value=float(default_lo), key=f"lo_{context}_{function_source}")
    with hi_col: x_hi = st.number_input("Analysis interval end", value=float(default_hi), key=f"hi_{context}_{function_source}")
    if x_lo >= x_hi:
        st.error("Analysis interval start must be less than interval end.")
    else:
        point = st.slider("Current input value", min_value=float(x_lo), max_value=float(x_hi), value=float((x_lo+x_hi)/2), key=f"point_{context}_{function_source}")
        st.session_state.setdefault("playback_point", float(point))
        ar_col1, ar_col2 = st.columns(2)
        with ar_col1: sec_a = st.slider("Average-rate interval start", min_value=float(x_lo), max_value=float(x_hi), value=float(x_lo), key=f"seca_{context}_{function_source}")
        with ar_col2: sec_b = st.slider("Average-rate interval end", min_value=float(x_lo), max_value=float(x_hi), value=float(x_hi), key=f"secb_{context}_{function_source}")
        show_f = st.checkbox("Show f(x)", True)
        show_d1 = st.checkbox("Show f′(x)", course == "Calculus")
        show_d2 = st.checkbox("Show f″(x)", course == "Calculus")
        show_tangent = st.checkbox("Show tangent at current point", True)
        show_secant = st.checkbox("Show secant over selected interval", True)
        speed = st.select_slider("Playback speed", options=["Slow", "Normal", "Fast"], value="Normal")
        play_col1, play_col2, play_col3 = st.columns(3)
        with play_col1: play = st.button("▶ Play one step")
        with play_col2: pause = st.button("Ⅱ Pause")
        with play_col3: reset = st.button("↺ Reset point")
        if reset:
            st.session_state[f"point_{context}_{function_source}"] = float((x_lo+x_hi)/2)
            st.rerun()
        if play:
            step = {"Slow": 0.02, "Normal": 0.05, "Fast": 0.12}[speed] * (x_hi-x_lo)
            next_point = point + step
            if next_point > x_hi: next_point = x_lo
            st.session_state[f"point_{context}_{function_source}"] = float(next_point)
            st.rerun()
        if pause:
            st.caption("Playback paused. “Play one step” advances the synchronized point once per click; this works reliably on hosted Streamlit without a background timer.")
        grid, ys, dys, ddys, critical, extrema, inflections = approximate_features(expr, x_lo, x_hi)
        fn, f1, f2 = numeric_fn(expr), numeric_fn(d1_expr), numeric_fn(d2_expr)
        ypoint = float(safe_eval(fn, point))
        slope = float(safe_eval(f1, point))
        curvature = float(safe_eval(f2, point))
        fig2, ax2 = plt.subplots(figsize=(10, 5.2))
        if show_f: ax2.plot(grid, ys, label="f(x)")
        if show_d1: ax2.plot(grid, dys, label="f′(x)")
        if show_d2: ax2.plot(grid, ddys, label="f″(x)")
        if show_f and np.isfinite(ypoint):
            ax2.scatter([point], [ypoint], s=65, marker="o", label=f"Current point x={point:.4g}")
        if show_tangent and show_f and show_d1 and np.isfinite(ypoint) and np.isfinite(slope):
            tx = np.linspace(max(x_lo, point-(x_hi-x_lo)*.15), min(x_hi, point+(x_hi-x_lo)*.15), 80)
            ax2.plot(tx, ypoint+slope*(tx-point), linestyle="--", label="Tangent line")
        if show_secant and show_f and sec_a != sec_b:
            ya, yb = float(safe_eval(fn, sec_a)), float(safe_eval(fn, sec_b))
            if np.isfinite(ya) and np.isfinite(yb):
                sec_slope = (yb-ya)/(sec_b-sec_a)
                sx = np.linspace(sec_a, sec_b, 80)
                ax2.plot(sx, ya+sec_slope*(sx-sec_a), linestyle=":", label=f"Secant (slope={sec_slope:.4g})")
        ax2.set(xlabel=f"{x_name} ({x_units})", ylabel=f"{y_name} ({y_units})", title=f"{context}: {description}")
        ax2.grid(True, alpha=.25); ax2.legend(fontsize=8); fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        if sec_a != sec_b:
            ya, yb = float(safe_eval(fn, sec_a)), float(safe_eval(fn, sec_b))
            sec_rate = (yb-ya)/(sec_b-sec_a)
            st.write(f"**Average rate of change:** ({yb:.6g} − {ya:.6g}) / ({sec_b:.6g} − {sec_a:.6g}) = **{sec_rate:.6g} {y_units}/{x_units}**.")
        else:
            st.write("Average rate of change is undefined when the interval endpoints are equal.")
        st.write(f"**At x = {point:.6g}:** f(x) = {ypoint:.6g} {y_units}; f′(x) = {slope:.6g} {y_units}/{x_units}; f″(x) = {curvature:.6g} {y_units}/{x_units}².")
        st.write(f"Interpretation: the function gives {y_name}; the first derivative is its instantaneous rate of change, and the second derivative describes how that rate changes. These interpretations depend on the units and context description supplied by the team.")
        if course == "Calculus":
            st.markdown("#### Numerical calculus analysis")
            st.caption("Critical points and inflection points below are numerical approximations over the selected interval. The app checks sign changes; tangential roots or very close features may be missed. Singularities are shown as gaps where non-finite values occur.")
            if critical:
                st.write("Zeros / candidates of f′(x): " + ", ".join(f"{v:.5g}" for v in critical))
            else: st.write("No numerical f′(x)=0 candidates detected on this interval.")
            if extrema:
                st.write("Local extrema verified by a first-derivative sign change:")
                st.dataframe(pd.DataFrame([{"x (approx.)": p, "f(x)": y, "classification": typ} for p,y,typ in extrema]), use_container_width=True)
            else: st.write("No local maxima/minima verified by a first-derivative sign change.")
            if inflections:
                st.write("Inflection points with detected concavity change:")
                st.dataframe(pd.DataFrame([{"x (approx.)": p, "f(x)": y} for p,y in inflections]), use_container_width=True)
            else: st.write("No inflection point with a detected concavity change found on this interval.")
            finite = np.isfinite(ys)
            if finite.any():
                imin, imax = np.nanargmin(ys), np.nanargmax(ys)
                st.write(f"**Approximate absolute extrema on closed interval:** min f({grid[imin]:.5g}) ≈ {ys[imin]:.6g}; max f({grid[imax]:.5g}) ≈ {ys[imax]:.6g}. Grid-based estimates; check endpoints and discontinuities.")
            st.write("**Increasing/decreasing:** determined approximately from the sign of f′(x) between candidate critical points. **Domain/discontinuities:** inspect denominators and logarithm/square-root restrictions in the entered formula; the graph leaves non-real/non-finite values blank. This is a numerical classroom aid, not a formal symbolic proof.")
        st.markdown("#### Contextual visual simulation")
        visual = st.selectbox("Visual simulation", ["Walking person with prosthetic leg", "Moving object / projectile", "Tank or drug concentration", "Population / cell colony", "Thermometer / weather", "Business / economic indicator", "Generic moving point"])
        # Map function output to a normalized 0..1 display; state explicitly this is visual scaling, not physical calibration.
        vals = ys[np.isfinite(ys)]
        if len(vals) and np.isfinite(ypoint):
            vmin, vmax = float(np.min(vals)), float(np.max(vals))
            norm = 0.5 if abs(vmax-vmin) < 1e-12 else float(np.clip((ypoint-vmin)/(vmax-vmin), 0, 1))
        else: norm = 0.5
        st.caption("Visual position/size is normalized to the finite function values on the selected interval (0–100%); this is illustrative scaling, not a calibrated physical model.")
        if visual == "Walking person with prosthetic leg":
            st.markdown(f"🧍‍♀️  🦿  **Gait phase:** {point:.3g} {x_units} • signal level: {norm*100:.1f}%")
            st.progress(norm)
        elif visual == "Moving object / projectile":
            st.markdown(f"🎯 Object position indicator: x={point:.3g}; normalized height/output={norm*100:.1f}%")
            st.progress(norm)
        elif visual == "Tank or drug concentration":
            st.markdown(f"🧪 Tank/concentration level: {norm*100:.1f}% of the displayed scale")
            st.progress(norm)
        elif visual == "Population / cell colony":
            st.markdown(f"🦠 Colony indicator: {norm*100:.1f}% of displayed scale")
            st.progress(norm)
        elif visual == "Thermometer / weather":
            st.markdown(f"🌡️ Temperature indicator: {norm*100:.1f}% of displayed scale")
            st.progress(norm)
        elif visual == "Business / economic indicator":
            st.markdown(f"📈 Indicator level: {ypoint:.5g} {y_units}; normalized display {norm*100:.1f}%")
            st.progress(norm)
        else:
            st.markdown(f"Moving point: x={point:.3g}, y={ypoint:.5g}")
            st.progress(norm)
        plt.close(fig2)
except ValueError as exc:
    st.error(str(exc))
except (TypeError, OverflowError, ZeroDivisionError) as exc:
    st.error(f"Function evaluation failed on the selected interval: {exc}. Check domain restrictions and parameters.")

st.subheader("6. Export the current work")
edited_flag = bool(st.session_state.get("data_edited", False))
export_data = validated.copy() if "validated" in locals() else pd.DataFrame(columns=["year","value"])
summary = {
    "Country": country_label, "Indicator": indicator_label, "Indicator code": indicator_code,
    "Units": units, "Requested range": f"{START_YEAR}–{END_YEAR}",
    "Actual coverage": f"{export_data.year.min()}–{export_data.year.max()}" if not export_data.empty else "No valid data",
    "Observation count": len(export_data), "Retrieval date": date.today().isoformat(),
    "Data status": "USER-MODIFIED SCENARIO" if edited_flag else "Source-matched working data",
    "Degree": degree, "Equation": equation if fit_ok else "Not fitted",
    "In-sample R²": f"{r2:.8g}" if fit_ok else "Unavailable",
    "In-sample MAE": f"{mae:.8g}" if fit_ok else "Unavailable",
    "Findings / limitations": "Polynomial fits describe selected observations, not causes. Extrapolation may be implausible; MAE/residuals describe in-sample fit."
}
csv_bytes = export_data.to_csv(index=False).encode("utf-8")
st.download_button("Download working data CSV", data=csv_bytes, file_name="hispanic_heritage_working_data.csv", mime="text/csv")
excel_buffer = io.BytesIO()
with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
    pd.DataFrame([{"item": k, "value": str(v)} for k,v in summary.items()]).to_excel(writer, sheet_name="Summary", index=False)
    export_data.to_excel(writer, sheet_name="Working data", index=False)
    if batch_result is not None and isinstance(batch_result, pd.DataFrame):
        batch_result.to_excel(writer, sheet_name="Calculated years", index=False)
    elif single_result:
        pd.DataFrame([single_result]).to_excel(writer, sheet_name="Calculated years", index=False)
    else:
        pd.DataFrame(columns=["year","estimate","units","classification"]).to_excel(writer, sheet_name="Calculated years", index=False)
st.download_button("Download Excel workbook (.xlsx)", data=excel_buffer.getvalue(), file_name="hispanic_heritage_explorer.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
if chart_png:
    pdf_bytes = make_pdf(summary, chart_png, export_data, batch_result if isinstance(batch_result, pd.DataFrame) else pd.DataFrame([single_result]) if single_result else pd.DataFrame())
    st.download_button("Download PDF report", data=pdf_bytes, file_name="hispanic_heritage_report.pdf", mime="application/pdf")
else:
    st.info("PDF report becomes available after a valid polynomial fit and chart are available.")

with st.expander("How to read and defend this project"):
    st.markdown("""
- **Data flow:** select country and indicator → request the official World Bank API → preserve the source series → edit a separate working table → validate → fit centered polynomial → calculate estimates → visualize and export.
- **Historical-data caveat:** missing observations remain missing. A country-indicator series may cover fewer than 70 years. Definitions, reporting methods, and country coverage can change.
- **Mathematical assumption:** polynomial least squares minimizes squared residuals on the selected historical points. A high degree can overfit; R² and MAE are in-sample summaries, not proof of predictive skill.
- **Holdout:** when enough years exist, the final chronological observations are held out from a fit to earlier years and scored separately with MAE.
- **Extrapolation:** polynomials can diverge rapidly outside the observed range. Predictions are not clipped and are not guaranteed forecasts.
- **Function lab:** formulas use Python/SymPy notation: `x**2`, `sin(x)`, `exp(x)`, `log(x)`, `(x+1)/(x-2)`. Use `*` for multiplication. Domain checks are numerical and do not replace a proof.
- **Course pathways:** Algebra 2 emphasizes graphs, domain/range, end behavior, and average rate; Precalculus emphasizes coefficient/degree effects and concavity; Calculus includes numerical derivative analysis.
- **Interpretation:** do not claim that a fitted curve proves historical causes. Explain associations, gaps, assumptions, units, and uncertainty.
""")
if printer_mode:
    st.markdown("---")
    st.markdown("### Printer-friendly report")
    st.write(f"{country_label} • {indicator_label} • {units} • {len(export_data)} valid observations")
    st.write(summary["Equation"])
    st.write(f"In-sample R²: {summary['In-sample R²']} • In-sample MAE: {summary['In-sample MAE']}")
    st.write(summary["Findings / limitations"])
