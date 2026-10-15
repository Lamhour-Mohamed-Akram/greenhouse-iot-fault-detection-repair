# Script 07 - Operating regimes and definitions used by the fault taxonomy
#
# Adds to the characterization of script 01:
#   - operating regimes of the deployment (sampling density, daily variability)
#   - variance collapse of the air-temperature channel after the long outage
#   - table of physical bounds and plausible-median bands with their sources
#   - dropout prevalence on the event-duration basis and on the grid-slot basis
#   - CO2 day/night statistics (raw-sample basis), and on the 1-min time basis per operating regime
#   - overlap between the longest constant run and the out-of-range episode
#   - prevalence figure with the whole-sensor convention drawn explicitly
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] if "__file__" in dir() else Path("..").resolve()
DATA = ROOT / "data" / "data.csv.gz"
RES = ROOT / "results"; OUT_T = RES / "tables"; OUT_F = RES / "figures"
TAB, INJ, MODELS, OFIG, ORIG_T, MS_F = OUT_T, RES / "injected", RES / "models", OUT_F, OUT_T, OUT_F
OUT_T.mkdir(parents=True, exist_ok=True); OUT_F.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight", "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})

SENSORS = ["humidity", "temperature", "soil_moisture", "soil_temp", "co2", "light"]
VALID_RANGE = {"humidity": (0, 100), "temperature": (-5, 55), "soil_moisture": (0, 100),
               "soil_temp": (-5, 55), "co2": (0, 5000), "light": (0, 1)}
PLAUSIBLE_MEDIAN = {"humidity": (20, 100), "temperature": (5, 45), "soil_moisture": (0, 100),
                    "soil_temp": (5, 45), "co2": (350, 2000), "light": (0, 1)}
UNITS = {"humidity": "% RH", "temperature": "degC", "soil_moisture": "index units (relative index, nominal range 0--100)", "soil_temp": "degC", "co2": "ppm", "light": "binary"}
SOURCE = {
    "humidity": ("DHT22 datasheet range 0--100 % RH", "greenhouse air is never drier than 20 % RH over a month (agronomic plausibility)"),
    "temperature": ("DHT22 datasheet -40--80 degC narrowed to the climatic envelope of Casablanca in spring", "monthly median air temperature of an unheated greenhouse"),
    "soil_moisture": ("nominal range of the firmware mapping of the ADC count between two fixed reference counts (not a calibrated volumetric scale); a value outside it means the probe output left the interval between the two reference counts, and is flagged", "none (relative index)"),
    "soil_temp": ("DS18B20 range narrowed to the climatic envelope", "monthly median soil temperature"),
    "co2": ("sensor cannot report negative ppm; 5000 ppm is the upper bound of enrichment practice", "long-run median between outdoor ambient and enrichment set points"),
    "light": ("binary digital output", "n/a"),
}

raw = pd.read_csv(DATA, encoding="latin-1")
raw.columns = ["Id"] + SENSORS + ["date"]
raw["date"] = pd.to_datetime(raw["date"]); raw = raw.sort_values("date").reset_index(drop=True)
g1 = raw.set_index("date")[SENSORS].resample("1min").median()

# ------------------------------------------------ bounds table
rows = []
for s in SENSORS:
    rows.append({"sensor": s, "unit": UNITS[s], "x_min": VALID_RANGE[s][0], "x_max": VALID_RANGE[s][1],
                 "mu_lo": PLAUSIBLE_MEDIAN[s][0], "mu_hi": PLAUSIBLE_MEDIAN[s][1],
                 "source_bounds": SOURCE[s][0], "source_median_band": SOURCE[s][1],
                 "observed_min": float(raw[s].min()), "observed_max": float(raw[s].max()), "observed_median": float(raw[s].median())})
pd.DataFrame(rows).to_csv(OUT_T / "bounds_table.csv", index=False)

# ------------------------------------------------ dropout bases
dt = raw["date"].diff().dt.total_seconds(); gi = np.where(dt > 300)[0]
gaps = pd.DataFrame({"start": raw["date"].iloc[gi - 1].values, "end": raw["date"].iloc[gi].values, "h": dt.iloc[gi].values / 3600})
span_h = (raw.date.max() - raw.date.min()).total_seconds() / 3600
miss_1m = g1["temperature"].isna()
dropout = {"events_gt5min": int(len(gaps)), "sum_event_hours": round(float(gaps.h.sum()), 1), "span_hours": round(span_h, 1),
           "event_basis_pct": round(100 * float(gaps.h.sum()) / span_h, 2),
           "grid_slot_basis_pct": round(100 * float(miss_1m.mean()), 2),
           "grid_slots_total": int(len(g1)), "grid_slots_missing": int(miss_1m.sum()),
           "note": "event basis sums the durations of the 15 gaps > 5 min; grid basis counts 1-min slots that received no sample (includes isolated missing minutes shorter than 5 min)"}

# ------------------------------------------------ CO2 day / night
co2 = raw.set_index("date")["co2"]; hr = co2.index.hour
co2_stats = {"median_all": round(float(co2.median()), 1), "median_night_22_05": round(float(co2[(hr >= 22) | (hr < 5)].median()), 1),
             "median_day_10_16": round(float(co2[(hr >= 10) & (hr < 16)].median()), 1),
             "p05": round(float(co2.quantile(.05)), 0), "p95": round(float(co2.quantile(.95)), 0),
             "pct_below_150": round(100 * float((co2 < 150).mean()), 1), "pct_below_420": round(100 * float((co2 < 420).mean()), 1),
             "daily_median_min": round(float(co2.resample("1D").median().min()), 0), "daily_median_max": round(float(co2.resample("1D").median().max()), 0),
             "n_days_daily_median_below_150": int((co2.resample("1D").median() < 150).sum())}

# ------------------------------------------------ 153.1 h constant run = OOR episode
s = raw["soil_moisture"]; d = raw["date"]; gap_break = d.diff().dt.total_seconds() > 300
g = ((s.diff() != 0) | gap_break).cumsum()
agg = pd.DataFrame({"v": s, "g": g, "d": d}).groupby("g").agg(v=("v", "first"), st=("d", "first"), en=("d", "last"), n=("d", "size"))
agg["h"] = (agg.en - agg.st).dt.total_seconds() / 3600
top = agg.sort_values("h", ascending=False).head(5).reset_index(drop=True)
top.to_csv(OUT_T / "longest_constant_runs_soil_moisture.csv", index=False)
longest = top.iloc[0]
stuck_live = agg[(agg.h > 1) & (agg.v >= 0) & (agg.v <= 100)]
overlap = {"longest_run_value": float(longest.v), "longest_run_hours": round(float(longest.h), 1), "longest_run_start": str(longest.st),
           "longest_run_end": str(longest.en), "longest_run_is_out_of_range": bool(longest.v < 0 or longest.v > 100),
           "longest_in_range_stuck_hours": round(float(stuck_live.h.max()), 2), "longest_in_range_stuck_start": str(stuck_live.loc[stuck_live.h.idxmax(), "st"])}

# ------------------------------------------------ regimes
daily = pd.DataFrame({
    "samples": raw.set_index("date")["Id"].resample("1D").count(),
    "median_dt_s": raw.set_index("date")["date"].diff().dt.total_seconds().resample("1D").median() if False else raw.set_index("date").index.to_series().diff().dt.total_seconds().resample("1D").median(),
    "temp_range": g1["temperature"].resample("1D").max() - g1["temperature"].resample("1D").min(),
    "temp_sd": g1["temperature"].resample("1D").std(),
    "temp_unique": g1["temperature"].resample("1D").nunique(),
    "hum_range": g1["humidity"].resample("1D").max() - g1["humidity"].resample("1D").min(),
    "hum_sd": g1["humidity"].resample("1D").std(),
    "sm_median": g1["soil_moisture"].resample("1D").median(),
    "sm_min": g1["soil_moisture"].resample("1D").min(), "sm_max": g1["soil_moisture"].resample("1D").max(),
    "minutes_with_data": g1["temperature"].resample("1D").count(),
}).round(2)
daily.to_csv(OUT_T / "daily_regime_profile.csv")

REGIMES = [("R1", "2022-04-18", "2022-04-27 12:33", "post-start, 1 sample/min, soil-moisture probe out of range"),
           ("R2", "2022-04-28 12:00", "2022-05-11 01:26", "dense sampling (about 1 sample/s), all dynamic channels varying"),
           ("R3", "2022-05-16 12:12", "2022-05-22 19:59", "after the 130.8-h outage: 1 sample/min, air-temperature variance collapse, soil-moisture level shift")]
reg_rows = []
for name, a, b, desc in REGIMES:
    r = raw[(raw.date >= a) & (raw.date <= b)]; gg = g1.loc[a:b]
    dd = daily.loc[pd.Timestamp(a).normalize():pd.Timestamp(b).normalize()]
    full = dd[dd.minutes_with_data >= 1000]   # days with (near) complete coverage
    reg_rows.append({"regime": name, "start": a, "end": b, "description": desc, "days": round((pd.Timestamp(b) - pd.Timestamp(a)).total_seconds() / 86400, 1),
                     "samples": int(len(r)), "median_dt_s": round(float(r.date.diff().dt.total_seconds().median()), 1),
                     "temp_daily_range_median_degC": round(float(full.temp_range.median()), 2),
                     "temp_daily_unique_median": round(float(full.temp_unique.median()), 1),
                     "temp_daily_sd_median": round(float(full.temp_sd.median()), 2),
                     "hum_daily_range_median": round(float(full.hum_range.median()), 1),
                     "sm_min": float(gg.soil_moisture.min()), "sm_max": float(gg.soil_moisture.max()), "sm_median": float(gg.soil_moisture.median()),
                     "n_full_days": int(len(full))})
reg = pd.DataFrame(reg_rows); reg.to_csv(OUT_T / "regimes.csv", index=False)
print(reg.to_string())

# variance collapse: days whose temperature daily range < 10 % of the R2 median daily range
r2_range = reg.loc[reg.regime == "R2", "temp_daily_range_median_degC"].iloc[0]
low = (daily.minutes_with_data >= 700) & (daily.temp_range < 0.1 * r2_range)
# variance collapse: at least three consecutive calendar days below the threshold
run_id = (low != low.shift()).cumsum(); run_len = low.groupby(run_id).transform("sum")
vc_days = daily[low & (run_len >= 3)]
vc = {"r2_median_daily_range_degC": float(r2_range), "threshold_degC": round(0.1 * float(r2_range), 2),
      "variance_collapse_days": [str(x.date()) for x in vc_days.index], "n_days": int(len(vc_days)),
      "pct_of_1min_slots_with_data": round(100 * float(sum(g1.loc[str(dday.date()), "temperature"].notna().sum() for dday in vc_days.index) / g1["temperature"].notna().sum()), 1) if len(vc_days) else 0.0,
      "pct_of_deployment_time": round(100 * float(len(vc_days) * 1440 / len(g1)), 1),
      "post_outage_days": [str(x.date()) for x in vc_days.index if x >= pd.Timestamp("2022-05-16")]}
print(vc)

# ------------------------------------------------ regime figure
fig, axes = plt.subplots(4, 1, figsize=(10, 7.5), sharex=True)
axes[0].bar(daily.index, daily.samples, width=0.8, color="tab:gray"); axes[0].set_yscale("log"); axes[0].set_ylabel("raw samples / day")
axes[1].plot(g1.index, g1.temperature, lw=0.4, color="tab:blue"); axes[1].set_ylabel("air temp. (degC)")
axes[2].plot(g1.index, g1.humidity, lw=0.4, color="tab:cyan"); axes[2].set_ylabel("air humidity (%)")
axes[3].plot(g1.index, g1.soil_moisture, lw=0.4, color="tab:brown"); axes[3].set_ylabel("soil-moisture index")
axes[3].axhspan(0, 100, color="tab:green", alpha=0.05)
for ax in axes:
    for _, gp in gaps[gaps.h > 12].iterrows(): ax.axvspan(gp.start, gp.end, color="gray", alpha=0.3)
    for name, a, b, _ in REGIMES:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color={"R1": "tab:red", "R2": "tab:green", "R3": "tab:orange"}[name], alpha=0.06)
for name, a, b, _ in REGIMES:
    axes[0].text(pd.Timestamp(a) + (pd.Timestamp(b) - pd.Timestamp(a)) / 2, daily.samples.max() * 1.6, name, ha="center", fontsize=9, weight="bold")
axes[0].set_ylim(top=daily.samples.max() * 6)
fig.autofmt_xdate(); fig.savefig(OUT_F / "fig_regimes.png"); fig.savefig(OUT_F / "fig_regimes.pdf"); plt.close(fig)

# ------------------------------------------------ prevalence figure with the whole-sensor convention
summary = pd.read_csv(ORIG_T / "fault_taxonomy_summary.csv")
plot_df = summary.set_index("sensor")[["dropout_pct", "oor_pct", "stuck_pct", "spike_pct"]]
ax = plot_df.plot(kind="bar", stacked=True, figsize=(7, 3.6), color=["gray", "tab:red", "tab:purple", "tab:orange"], fontsize=10)
ax.set_ylabel("% of deployment time (1-min basis)", fontsize=10); ax.set_xlabel(""); ax.tick_params(axis="x", labelrotation=0, labelsize=9)
ax.legend(["Dropout", "Out-of-range", "Stuck-at", "Spike"], fontsize=9, loc="upper left")
ax.set_ylim(0, 122); ax.set_yticks(range(0, 101, 20))
ax.set_xticklabels(["Air\nhumidity", "Air\ntemperature", "Soil-moisture\nindex", "Soil\ntemperature", "CO$_2$", "Light"][:len(plot_df)])
for i, s in enumerate(summary.sensor):
    row = summary.loc[summary.sensor == s].iloc[0]
    tag = "Zero\nvariance" if row.dead_sensor else ("Suspected\nbias" if row.calibration_bias else None)
    if tag:
        ax.plot([i - 0.4, i + 0.4], [100, 100], color="tab:red", lw=1.5)
        ax.annotate(tag, (i, 102), ha="center", va="bottom", color="tab:red", fontsize=9, weight="bold")
ax.set_title("Naturally occurring fault classes per sensor (red line: whole-sensor flag = 100 %)", fontsize=10)
plt.gcf().savefig(OUT_F / "fig2_fault_prevalence.png"); plt.gcf().savefig(OUT_F / "fig2_fault_prevalence.pdf"); plt.close()

# ------------------------------------------------ flag-definition numbers : online vs retrospective stuck labelling
# share of stuck-labelled minutes that lie in the first 60 min of their episode (would be missed by an online rule)
stk = agg[(agg.h > 1) & (agg.v >= 0) & (agg.v <= 100)]   # in-range stuck runs only (OOR runs belong to the OOR class)
first_hour = float((np.minimum(stk.h, 1.0)).sum() / stk.h.sum()) if len(stk) else 0.0
flagdef = {"stuck_episodes_gt1h_soil_moisture": int(len(stk)), "share_of_stuck_time_in_first_60min": round(100 * first_hour, 1),
           "note": "retrospective labelling marks the whole constant run; an online detector can flag only after minute 60, so point-wise recall of an online rule is bounded by 1 - share_of_first_60min"}

# ------------------------------------------------ CO2 on the 1-min time basis and per regime
# the statistics above are raw-sample weighted, so the 1-s regime R2 dominates them; the paper otherwise works on a time basis
c1 = g1["co2"].dropna(); h1 = c1.index.hour
co2_stats["basis"] = "raw samples (raw-sample weighted)"
co2_stats["time_basis_1min"] = {"median": round(float(c1.median()), 1), "pct_below_150": round(100 * float((c1 < 150).mean()), 1),
                                "median_night_22_05": round(float(c1[(h1 >= 22) | (h1 < 5)].median()), 1),
                                "median_day_10_16": round(float(c1[(h1 >= 10) & (h1 < 16)].median()), 1)}
co2_stats["per_regime"] = {}
for name, a, b, _ in REGIMES:
    r = co2[(co2.index >= a) & (co2.index <= b)]; m = c1.loc[a:b]; hm = m.index.hour
    co2_stats["per_regime"][name] = {"raw_sample_share_pct": round(100 * len(r) / len(co2), 1), "median_1min": round(float(m.median()), 1),
                                     "median_night_22_05_1min": round(float(m[(hm >= 22) | (hm < 5)].median()), 1),
                                     "median_day_10_16_1min": round(float(m[(hm >= 10) & (hm < 16)].median()), 1),
                                     "pct_below_150_1min": round(100 * float((m < 150).mean()), 1)}

out = {"dropout": dropout, "co2": co2_stats, "longest_constant_run": overlap, "variance_collapse": vc, "flag_definitions": flagdef}
with open(OUT_T / "characterization_extended_headline.json", "w") as f: json.dump(out, f, indent=2, default=str)
print(json.dumps(out, indent=2, default=str)); print("done")
