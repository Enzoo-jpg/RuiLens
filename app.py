"""
瑞视 · 乌帕替尼(瑞福) DTP 销售运营看板 (RuiLens v2.9)
==================================================
已确认口径：
- 患者唯一键 = oneid + 开票抬头（两者皆空时回退 会员号 / 会员姓名）
- 首次购药时间 = 每个患者最小的 销售时间
- 当月新患 = 首次购药时间落在该统计月的患者数
- YTD 老患 = 自 YTD 开始日起窗口内有购药记录的患者滚动累积（不再卡首购月，首购早但期内复购者从复购月起入池）；当月新患 = 全局首购月 == 当月；YTD老患 = 累计患者 - 当月新患。
- 城市 = 药房名称前 2 字
- 适应症 = 每个患者末次(最近)购药的适应症，经清洗映射表归到标准适应症
- 分布图当月患者 = 当月有销售记录的患者（按末次购药时的药房/适应症归类）

v2.2：新增「药房维度」Tab，含累计OP、活跃/OP/NP、复购率（末次购药+30天推算）、R12M DOT 及环比。
"""

import io
import pandas as pd
import numpy as np
from datetime import date
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# 映射表（非隐私，随仓库发布；可在侧边栏上传覆盖）
DEFAULT_MAP_PATH = "assets/indication_mapping.xlsx"


# ------------------------- 数据加载 -------------------------
def load_data(uploaded_file, path):
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, sheet_name=0)
    else:
        df = pd.read_excel(path, sheet_name=0)
    df.columns = [str(c).strip() for c in df.columns]
    df["销售时间"] = pd.to_datetime(df["销售时间"], errors="coerce")
    return df


# ------------------------- 隐私列剔除 -------------------------
# 直接标识符列（电话/姓名/会员号）不参与任何计算，仅用于 make_key 兜底；
# 置空后内存中不再含敏感值，且 make_key 调用不会因列缺失而报错。
PII_DROP_SUBSTRINGS = ["电话", "姓名", "会员号"]


def strip_pii_columns(df):
    """将直接标识符列（电话/姓名/会员号）置空，返回 (df, 被剔除的列名列表)。
    列名保留（值为 NA），以保证 make_key 兜底逻辑可用且不破坏索引。"""
    dropped = [c for c in df.columns if any(s in c for s in PII_DROP_SUBSTRINGS)]
    for c in dropped:
        df[c] = pd.NA
    return df, dropped


# ------------------------- 患者键 -------------------------
def make_key(oid, title, mem, name):
    parts = []
    if pd.notna(oid) and str(oid).strip() != "":
        parts.append(str(oid).strip())
    if pd.notna(title) and str(title).strip() != "":
        parts.append(str(title).strip())
    if not parts:  # 两者皆空 -> 回退
        for v in (mem, name):
            if pd.notna(v) and str(v).strip() != "":
                parts.append(str(v).strip())
                break
    return "|".join(parts) if parts else None


def build_patient_table(df):
    d = df.copy()
    d["key"] = [
        make_key(o, t, m, n)
        for o, t, m, n in zip(d["oneid"], d["开票抬头"], d["会员号"], d["会员姓名"])
    ]
    d = d[d["key"].notna()].copy()
    d = d.sort_values("销售时间")
    first = d.groupby("key")["销售时间"].min().rename("首次购药时间")
    last = d.groupby("key")["销售时间"].max().rename("末次购药时间")
    # 末次(最近)购药行的适应症与药房
    last_rows = d.groupby("key").tail(1).set_index("key")
    last_ind = last_rows["适应症"].rename("末次适应症")
    last_pharm = last_rows["药房名称"].rename("末次药房")
    pt = pd.concat([first, last, last_ind, last_pharm], axis=1).reset_index()
    pt["首购月"] = pt["首次购药时间"].dt.to_period("M").dt.to_timestamp()
    return d, pt


# ------------------------- 患者池累计 -------------------------
def patient_pool(d_keyed, pt, ytd_start, disp_start, disp_end):
    """患者池累计（v2.10 口径，按用户纠正）：
    - 累积患者 = 自 YTD 开始日起，截至当月，窗口内「有购药记录」的去重患者滚动累积。
      入选条件不再卡首购月（首购早于 YTD、但期内复购的患者，从复购月起即入池）。
      断药的历史患者仍留在累计池（他曾在窗口内购药）。
    - 当月新患 = 全局首购月 == 当月。
    - YTD老患 = 累计患者 - 当月新患（= 当月活跃患者数 - 当月新患）。"""
    ytd_start_ts = pd.Timestamp(ytd_start).to_period("M").to_timestamp()
    all_months = pd.period_range(ytd_start_ts, pd.Timestamp(disp_end), freq="M").to_timestamp()
    dk = d_keyed[d_keyed["销售时间"] >= ytd_start_ts].copy()
    dk["月"] = dk["销售时间"].dt.to_period("M").dt.to_timestamp()
    first_month = pt.set_index("key")["首购月"]
    rows = []
    cum_set = set()
    for m in all_months:
        active = set(dk.loc[dk["月"] == m, "key"].unique())
        new = {k for k in active if first_month.get(k) == m}
        cum_set |= active
        rows.append({
            "月份": m,
            "当月活跃患者": len(active),
            "当月新患": len(new),
            "当月老患(复购)": len(active) - len(new),
            "YTD老患": len(cum_set) - len(new),
            "累计患者": len(cum_set),
        })
    out = pd.DataFrame(rows)
    out = out[(out["月份"] >= pd.Timestamp(disp_start)) & (out["月份"] <= pd.Timestamp(disp_end))]
    return out


# ------------------------- 当月分布 -------------------------
def _month_bounds(target_month):
    m_start = pd.Timestamp(target_month).to_period("M").to_timestamp()
    # 月末 24 点（含当月最后一天全天），避免 MonthEnd 零点漏算月末销售
    m_end = pd.Timestamp(target_month).to_period("M").to_timestamp(how="end")
    return m_start, m_end


def pharmacy_dist(df, target_month):
    m_start, m_end = _month_bounds(target_month)
    sub = df[(df["销售时间"] >= m_start) & (df["销售时间"] <= m_end)].sort_values("销售时间")
    # 每个活跃患者归到当月“最后一次购买”的药房（保证饼图占比合计 100%）
    last_pharm = sub.groupby("key")["药房名称"].tail(1)
    return last_pharm.value_counts()


def indication_dist(df, pt, target_month, mapping):
    m_start, m_end = _month_bounds(target_month)
    active_keys = df[(df["销售时间"] >= m_start) & (df["销售时间"] <= m_end)]["key"].unique()
    sub_pt = pt[pt["key"].isin(active_keys)]
    raw = sub_pt["末次适应症"]
    raw_str = raw.apply(lambda x: None if pd.isna(x) else str(x).strip())
    if mapping is not None:
        mapped = raw_str.map(mapping)
        # 空值(NaN 原始适应症)一律自动归为“其他”
        mapped = mapped.where(raw_str.notna(), "其他")
        mapped = mapped.fillna("其他(未映射)")
    else:
        # 无映射表：空值也归“其他”，其余保留原始写法
        mapped = raw_str.apply(lambda x: "其他" if x is None else x)
    counts = mapped.value_counts()
    # 未映射明细：仅非空且映射表里找不到的原始写法（供网页补填维护）
    if mapping is not None:
        unmapped_mask = raw_str.notna() & raw_str.map(mapping).isna()
        unmapped = raw[unmapped_mask].astype(str).value_counts().rename_axis("原始适应症").reset_index(name="患者数")
    else:
        unmapped = raw.dropna().astype(str).value_counts().rename_axis("原始适应症").reset_index(name="患者数")
    return counts, unmapped


# ------------------------- 映射表读取 -------------------------
def find_col(cols, *cands):
    """在列名列表 cols 中按候选子串查找列名，找不到返回 None。"""
    for c in cols:
        if any(cand in c for cand in cands):
            return c
    return None


def load_mapping(uploaded_file=None, path=None):
    """读取 4 列清洗映射表（末次适应症 / TA / 末次适应症（清洗） / 规范治疗数量）。
    返回 (raw->clean 的映射字典, 原始映射 DataFrame)。读取失败或无源返回 (None, None)。"""
    src = uploaded_file if uploaded_file is not None else path
    if src is None:
        return None, None
    try:
        mp = pd.read_excel(src, sheet_name=0)
    except Exception as e:
        st.error(f"读取映射表失败：{e}")
        return None, None
    mp.columns = [str(c).strip() for c in mp.columns]
    raw_col = find_col(mp.columns, "末次适应症", "原始适应症", "适应症")
    clean_col = find_col(mp.columns, "清洗", "标准", "归一")
    if raw_col is None or clean_col is None:
        st.error("映射表需包含「末次适应症」与「末次适应症（清洗）」两列")
        return None, None
    # 同一原始写法去重，取首次
    mp = mp.drop_duplicates(subset=[raw_col], keep="first")
    # 构造映射；空原始(NaN)归一为占位键，匹配映射表中 None->其他 的行
    mapping = {}
    for _, row in mp.iterrows():
        raw = row[raw_col]
        key = "（空）" if pd.isna(raw) else str(raw).strip()
        mapping[key] = str(row[clean_col]).strip()
    return mapping, mp


def std_qty_from_map(mp_df):
    """从映射表提取 标准适应症 -> 规范治疗数量（"无"/空 -> None，表示不计入）。"""
    out = {}
    if mp_df is None:
        return out
    std_col = find_col(mp_df.columns, "规范治疗数量", "规范治疗", "治疗数量")
    clean_col = find_col(mp_df.columns, "清洗", "标准", "归一")
    if std_col is None or clean_col is None:
        return out
    for _, row in mp_df.iterrows():
        c = str(row[clean_col]).strip()
        v = row[std_col]
        if pd.isna(v) or str(v).strip() in ("", "无"):
            out[c] = None
        else:
            try:
                out[c] = float(v)
            except ValueError:
                out[c] = None
    return out


# ------------------------- 趋势分析计算 -------------------------
def _clean_indication(series, mapping):
    """原始适应症 -> 清洗后标准值；空值->其他；找不到->其他(未映射)。"""
    raw = series.apply(lambda x: None if pd.isna(x) else str(x).strip())
    if mapping is not None:
        mapped = raw.map(mapping)
        mapped = mapped.where(raw.notna(), "其他")
        mapped = mapped.fillna("其他(未映射)")
    else:
        mapped = raw.apply(lambda x: "其他" if x is None else x)
    return mapped


def sales_trend(d, pt, disp_start, disp_end):
    """每月 新患销量 / 老患销量（净销量盒数）。"""
    d = d.copy()
    d["销售数量"] = pd.to_numeric(d["销售数量"], errors="coerce").fillna(0)
    d["月"] = d["销售时间"].dt.to_period("M").dt.to_timestamp()
    first_month = pt.set_index("key")["首购月"]
    months = pd.period_range(pd.Timestamp(disp_start), pd.Timestamp(disp_end), freq="M").to_timestamp()
    rows = []
    for m in months:
        sub = d[d["月"] == m]
        if len(sub) == 0:
            rows.append({"月份": m, "新患销量": 0.0, "老患销量": 0.0})
            continue
        fm = sub["key"].map(first_month)
        is_new = (fm == m)
        rows.append({
            "月份": m,
            "新患销量": float(sub.loc[is_new, "销售数量"].sum()),
            "老患销量": float(sub.loc[~is_new, "销售数量"].sum()),
        })
    return pd.DataFrame(rows)


def np_by_indication(d, pt, disp_start, disp_end, mapping):
    """展示窗口内每月各适应症的 新患数（首购月 == 月）。返回长表。"""
    new_pt = pt.copy()
    new_pt["首购月"] = new_pt["首次购药时间"].dt.to_period("M").dt.to_timestamp()
    new_pt["适应症"] = _clean_indication(new_pt["末次适应症"], mapping)
    win = new_pt[(new_pt["首购月"] >= pd.Timestamp(disp_start)) &
                 (new_pt["首购月"] <= pd.Timestamp(disp_end))]
    if len(win) == 0:
        return pd.DataFrame(columns=["月份", "适应症", "新患数"])
    g = win.groupby(["首购月", "适应症"])["key"].nunique().reset_index()
    g = g.rename(columns={"首购月": "月份", "key": "新患数"})
    return g


def treatment_rate(d, pt, target_month, mapping, std_qty):
    """规范治疗率：去除当月新患，老患者从首次购药到 target_month 月末累计盒数达标占比。
    无规范治疗量（其他/未映射）的适应症不计入。"""
    t_start, t_end = _month_bounds(target_month)
    d = d.copy()
    d["销售数量"] = pd.to_numeric(d["销售数量"], errors="coerce").fillna(0)
    d_sub = d[d["销售时间"] <= t_end]
    old_pt = pt[pt["首次购药时间"] < t_start].copy()  # 老患者：首次购药早于当月1日
    if len(old_pt) == 0:
        return pd.DataFrame(columns=["适应症", "老患者数", "达标患者数", "达标率"])
    cum = d_sub.groupby("key")["销售数量"].sum()
    old_pt["累计盒数"] = old_pt["key"].map(cum).fillna(0)
    old_pt["适应症"] = _clean_indication(old_pt["末次适应症"], mapping)
    rows = []
    for ind, g in old_pt.groupby("适应症"):
        std = std_qty.get(ind)
        if std is None:
            continue
        hit = int((g["累计盒数"] >= std).sum())
        rows.append({
            "适应症": ind,
            "老患者数": len(g),
            "达标患者数": hit,
            "达标率": hit / len(g),
        })
    return pd.DataFrame(rows)


def r12m_dot(d, pt, disp_start, disp_end, mapping):
    """每月回滚12个月 DOT = 滚动12月净销量盒数总和 / 滚动12月去重患者数，分适应症。返回长表。"""
    d = d.copy()
    d["销售数量"] = pd.to_numeric(d["销售数量"], errors="coerce").fillna(0)
    last_ind = pt.set_index("key")["末次适应症"]
    clean_series = _clean_indication(last_ind, mapping)
    d["适应症"] = d["key"].map(clean_series)
    months = pd.period_range(pd.Timestamp(disp_start), pd.Timestamp(disp_end), freq="M").to_timestamp()
    recs = []
    for m in months:
        start = (pd.Timestamp(m).to_period("M") - 11).to_timestamp()
        end = pd.Timestamp(m).to_period("M").to_timestamp() + pd.offsets.MonthEnd(1)
        w = d[(d["销售时间"] >= start) & (d["销售时间"] <= end)]
        if len(w) == 0:
            continue
        for ind, g in w.groupby("适应症"):
            uniq = g["key"].nunique()
            recs.append({
                "月份": m,
                "适应症": ind,
                "DOT": float(g["销售数量"].sum() / uniq) if uniq else 0.0,
            })
    return pd.DataFrame(recs)


# ------------------------- 药房维度计算 -------------------------
def pharmacy_table(d, pt, ytd_start, target_month):
    """药房维度汇总表：累计OP、目标月活跃/OP/NP、复购率（末次购药+30天）、R12M DOT 及环比。

    口径：
    - 患者按每笔销售记录归属药房，一个患者可在多个药房出现。
    - 累计OP：窗口 [YTD 开始日, 目标月末] 内在该药房有购买、且首购早于目标月的老患者（去重）。
      即用户透视表口径：YTD~目标月末 在该药房有销量的患者去重，再剔除目标月新患。
    - 目标月活跃/OP/NP：目标月在该药房有购买的患者；OP 为首购早于目标月，NP 为首购在目标月。
    - 复购率：应购患者中目标月实际复购的占比。
        * 应购 = 首购早于目标月，且在该药房末次正销量购药 + 盒数x30 天落在目标月内。
        * 实际复购 = 应购患者中目标月在该药房有购买的患者。
    - R12M DOT：回滚 12 个月净销量盒数 / 去重患者数（药房维度）。
    - 环比：本月值 - 上月值（差值，非增长率）。
    """
    target_start, target_end = _month_bounds(target_month)
    prev_start, prev_end = _month_bounds(
        (pd.Timestamp(target_start).to_period("M") - 1).to_timestamp()
    )
    ytd_start_ts = pd.Timestamp(ytd_start).to_period("M").to_timestamp()
    first_purchase = pt.set_index("key")["首次购药时间"]

    def _month_metrics(m_start, m_end):
        # 累计OP：窗口 [ytd_start, 目标月末] 内在该药房有购买、且首购早于目标月的老患者（去重）
        # 即用户透视表口径：YTD~目标月末 在该药房有销量的患者去重，再剔除目标月新患
        win = d[(d["销售时间"] >= ytd_start_ts) & (d["销售时间"] <= m_end)].copy()
        cum_op = win[win["key"].map(first_purchase) < m_start].groupby("药房名称")["key"].nunique()

        # 目标月活跃/OP/NP
        sales_m = d[(d["销售时间"] >= m_start) & (d["销售时间"] <= m_end)].copy()
        active = sales_m.groupby("药房名称")["key"].nunique()
        op = sales_m[sales_m["key"].map(first_purchase) < m_start].groupby("药房名称")["key"].nunique()
        np = sales_m[sales_m["key"].map(first_purchase).between(m_start, m_end)].groupby("药房名称")["key"].nunique()

        # 复购率：应购 = 末次正销量购药 + 盒数*30 落在目标月
        sales_before = d[d["销售时间"] < m_start].copy()
        sales_before["销售数量"] = pd.to_numeric(sales_before["销售数量"], errors="coerce").fillna(0)
        sales_before = sales_before[sales_before["销售数量"] > 0]
        if len(sales_before):
            last_p = sales_before.sort_values("销售时间").groupby(["药房名称", "key"]).tail(1).copy()
            last_p["expected"] = last_p["销售时间"] + last_p["销售数量"] * pd.Timedelta(days=30)
            should_df = last_p[(last_p["expected"] >= m_start) & (last_p["expected"] <= m_end)]
            should = should_df.groupby("药房名称")["key"].nunique()
            active_keys = sales_m[["药房名称", "key"]].drop_duplicates()
            actual_df = should_df[["药房名称", "key"]].merge(
                active_keys, on=["药房名称", "key"], how="inner"
            )
            actual = actual_df.groupby("药房名称")["key"].nunique()
        else:
            should = pd.Series(dtype=float)
            actual = pd.Series(dtype=float)

        # R12M DOT
        dot_start = (pd.Timestamp(m_start).to_period("M") - 11).to_timestamp()
        sales_12m = d[(d["销售时间"] >= dot_start) & (d["销售时间"] <= m_end)].copy()
        sales_12m["销售数量"] = pd.to_numeric(sales_12m["销售数量"], errors="coerce").fillna(0)
        dot = sales_12m.groupby("药房名称").apply(
            lambda g: float(g["销售数量"].sum() / g["key"].nunique()) if g["key"].nunique() else 0.0
        )

        # 合并
        all_pharm = sorted(d["药房名称"].dropna().unique())
        df = pd.DataFrame(index=all_pharm)
        df.index.name = "药房名称"
        df["累计OP数量"] = cum_op
        df["目标月活跃人数"] = active
        df["目标月OP人数"] = op
        df["目标月NP人数"] = np
        df["应购患者数"] = should
        df["实际复购患者数"] = actual
        df["目标月复购率"] = actual / should
        df["R12M DOT"] = dot
        df = df.reset_index()
        df = df.fillna(0)
        df.loc[df["应购患者数"] == 0, "目标月复购率"] = None
        return df

    cur = _month_metrics(target_start, target_end)
    prev = _month_metrics(prev_start, prev_end)

    prev_cols = prev[["药房名称", "目标月复购率", "R12M DOT"]].rename(
        columns={"目标月复购率": "上月复购率", "R12M DOT": "上月DOT"}
    )
    merged = cur.merge(prev_cols, on="药房名称", how="left")
    merged["复购率环比"] = merged["目标月复购率"] - merged["上月复购率"]
    merged["R12M DOT环比"] = merged["R12M DOT"] - merged["上月DOT"]

    out = merged[
        ["药房名称", "累计OP数量", "目标月活跃人数", "目标月OP人数", "目标月NP人数",
         "应购患者数", "实际复购患者数", "目标月复购率", "复购率环比",
         "R12M DOT", "R12M DOT环比"]
    ].sort_values("累计OP数量", ascending=False).reset_index(drop=True)
    return out


# ------------------------- 单药房月度分析 -------------------------
_IND_ORDER = ["AD", "RA", "PsA", "UC", "CD", "其他"]


def single_pharmacy_monthly(d_all, pt_all, pharm_name, m_start, m_end, mapping):
    """单药房月度分析：适应症分布（按末次购药适应症）+ 新老患净销量分布。

    口径：
    - 数据范围：d_all 中药房名称 == pharm_name 且销售时间在 [m_start, m_end] 的记录。
    - 适应症分布：窗口内每月在该药房有购买的患者，按各自「末次购药适应症」（经映射）堆叠统计患者数。
    - 新老患净销量：每月净销量（同月正负抵消）按患者「全局首次购药月」拆分
      （新患 = 首购落在该月；老患 = 首购早于该月）。
    - 适应症映射：空值 / 未映射 → 「其他」；mapping 为有效映射字典（网页+文件合并）。
    """
    first_purchase = pt_all.set_index("key")["首次购药时间"]
    last_ind = pt_all.set_index("key")["末次适应症"]

    # 确保 d_all 带患者 key（df_all 为原始数据，可能未建 key）
    if "key" not in d_all.columns:
        d_all = d_all.copy()
        d_all["key"] = [
            make_key(o, t, m, n)
            for o, t, m, n in zip(d_all["oneid"], d_all["开票抬头"], d_all["会员号"], d_all["会员姓名"])
        ]
        d_all = d_all[d_all["key"].notna()].copy()

    def _map_ind(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return "其他"
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return "其他"
        return mapping.get(s, "其他") if mapping else "其他"

    sub = d_all[d_all["药房名称"].astype(str).str.strip() == pharm_name].copy()
    sub["销售数量"] = pd.to_numeric(sub["销售数量"], errors="coerce").fillna(0)
    m_start_ts = pd.Timestamp(m_start).to_period("M").to_timestamp()           # 起始月月初
    m_end_ts = pd.Timestamp(m_end).to_period("M").to_timestamp(how="end")      # 结束月月末
    sub = sub[(sub["销售时间"] >= m_start_ts) & (sub["销售时间"] <= m_end_ts)].copy()
    sub["月"] = sub["销售时间"].dt.to_period("M")

    months = pd.period_range(pd.Timestamp(m_start).to_period("M"), pd.Timestamp(m_end).to_period("M"), freq="M")

    # ---- 适应症分布（每月患者数，按末次购药适应症）----
    ind_rows = []
    for mk in months:
        msk = sub["月"] == mk
        keys = sub.loc[msk, "key"].unique()
        if len(keys) == 0:
            continue
        inds = last_ind.reindex(pd.Index(keys)).map(_map_ind)
        vc = inds.value_counts()
        for ind_name, cnt in vc.items():
            ind_rows.append({"月份": mk.to_timestamp(), "适应症": ind_name, "患者数": int(cnt)})
    ind_df = pd.DataFrame(ind_rows)

    # ---- 新老患净销量（按月，按全局首购月拆分）----
    if len(sub):
        g = sub.groupby(["月", "key"])["销售数量"].sum().reset_index()
        g["首购月"] = g["key"].map(first_purchase).dt.to_period("M")
        g["类型"] = np.where(g["首购月"] == g["月"], "新患", "老患")
        sales_pivot = g.groupby(["月", "类型"])["销售数量"].sum().unstack(fill_value=0)
        for c in ("老患", "新患"):
            if c not in sales_pivot.columns:
                sales_pivot[c] = 0
        sales_pivot = sales_pivot[["老患", "新患"]].reset_index()
        sales_pivot["月份"] = sales_pivot["月"].dt.to_timestamp()
        sales_df = sales_pivot.rename(columns={"老患": "老患销量", "新患": "新患销量"})[
            ["月份", "老患销量", "新患销量"]
        ]
    else:
        sales_df = pd.DataFrame(columns=["月份", "老患销量", "新患销量"])

    return ind_df, sales_df


def single_pharmacy_np(d_pharm, pt_all, m_start, m_end, mapping):
    """单药房、按末次购药适应症拆分的新患趋势（全局首次购药月 == 月）。
    返回长表 [月份, 适应症, 新患数]。"""
    pharm_keys = set(d_pharm["key"].unique())
    new_pt = pt_all[pt_all["key"].isin(pharm_keys)].copy()
    new_pt["首购月"] = new_pt["首次购药时间"].dt.to_period("M").dt.to_timestamp()
    new_pt["适应症"] = _clean_indication(new_pt["末次适应症"], mapping)
    win = new_pt[
        (new_pt["首购月"] >= pd.Timestamp(m_start)) &
        (new_pt["首购月"] <= pd.Timestamp(m_end))
    ]
    if len(win) == 0:
        return pd.DataFrame(columns=["月份", "适应症", "新患数"])
    g = win.groupby(["首购月", "适应症"])["key"].nunique().reset_index()
    g = g.rename(columns={"首购月": "月份", "key": "新患数"})
    return g


def single_pharmacy_treatment_rate(d_pharm, pt_all, m_start, m_end, mapping, std_qty):
    """单药房逐月规范治疗率（分适应症），口径同 treatment_rate 但仅限该药房患者：
    去除计算月新患（首购早于当月1日且在该药房有购药），看当月产生购药的患者里，
    既往历史（<=目标月末）购药盒数总和是否达到该适应症标准规范治疗量。
    返回长表 [月份, 适应症, 老患者数, 达标患者数, 未达标患者数, 达标率]。"""
    pharm_keys = pd.Index(d_pharm["key"].unique())
    months = pd.period_range(
        pd.Timestamp(m_start).to_period("M"),
        pd.Timestamp(m_end).to_period("M"), freq="M"
    ).to_timestamp()
    recs = []
    for m in months:
        t_start, t_end = _month_bounds(m)
        d_sub = d_pharm[d_pharm["销售时间"] <= t_end]
        old_pt = pt_all[
            pt_all["key"].isin(pharm_keys) & (pt_all["首次购药时间"] < t_start)
        ].copy()
        if len(old_pt) == 0:
            continue
        cum = d_sub.groupby("key")["销售数量"].sum()
        old_pt["累计盒数"] = old_pt["key"].map(cum).fillna(0)
        old_pt["适应症"] = _clean_indication(old_pt["末次适应症"], mapping)
        for ind, g in old_pt.groupby("适应症"):
            std = std_qty.get(ind)
            if std is None:
                continue
            hit = int((g["累计盒数"] >= std).sum())
            recs.append({
                "月份": m,
                "适应症": ind,
                "老患者数": len(g),
                "达标患者数": hit,
                "未达标患者数": len(g) - hit,
                "达标率": hit / len(g),
            })
    return pd.DataFrame(recs)


# ------------------------- 临床分析：TOP5 医院 -------------------------
def _mask_name(name):
    """医生姓名脱敏：保留首字，其余用 *；空 / nan 返回 None。"""
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return None
    s = str(name).strip()
    if s == "" or s.lower() == "nan":
        return None
    if len(s) == 1:
        return s + "*"
    return s[0] + "*" * (len(s) - 1)


def clinical_top5(df_all, pt_all, target_month, mapping):
    """临床分析 TOP5 医院（字段 = 医疗单位）。

    口径：
    - TOP5 = 以 target_month 为锚点的「最近3个月」(target-2, target-1, target) 累计净销量排名取前5。
    - 滚动3个月各列：销量 = 净销量盒数；新患 = 全局首购月 == 统计月 且 该院当净销量 > 0；
      老患 = 同院净销量 > 0 且 全局首购早于统计月。
    - R12M DOT = 回滚12个月(到 target 月末) 净销量 / 去重患者数。
    - 25年 DOT = 2025 全年净销量 / 去重患者数（固定基线，不随筛选变化）。
    - 同比 = (R12M DOT − 25年DOT) / 25年DOT。
    - 医生明细 = 每家 TOP3 医生（按 target 月销量排序），近两个月(锚定前1月、锚定月)
      销量 / 患者数 / 环比(增长率)，姓名脱敏。
    返回 (hosp_df, doctor_df)
    """
    d = df_all.copy()
    if "key" not in d.columns:
        d["key"] = [make_key(o, t, m, n) for o, t, m, n in
                    zip(d["oneid"], d["开票抬头"], d["会员号"], d["会员姓名"])]
        d = d[d["key"].notna()].copy()
    d["销售数量"] = pd.to_numeric(d["销售数量"], errors="coerce").fillna(0)
    d["医疗单位"] = d["医疗单位"].astype(str).str.strip()
    d = d[d["医疗单位"] != ""]
    first_purchase = pt_all.set_index("key")["首次购药时间"]

    target_p = pd.Timestamp(target_month).to_period("M")
    months3 = [target_p - 2, target_p - 1, target_p]
    month_labels = [m.to_timestamp().strftime("%Y-%m") for m in months3]

    # 预计算每个月 每院每 key 净销量
    month_net = {}
    for m in months3:
        ms, me = m.to_timestamp(), m.to_timestamp(how="end")
        sub = d[(d["销售时间"] >= ms) & (d["销售时间"] <= me)]
        month_net[str(m)] = sub.groupby(["医疗单位", "key"])["销售数量"].sum() if len(sub) else None

    # 候选医院 & 累计销量
    hosp_set, hosp_sales = set(), {}
    for g in month_net.values():
        if g is None:
            continue
        for h in g.index.get_level_values(0).unique():
            hosp_set.add(h)
    for h in hosp_set:
        tot = 0.0
        for g in month_net.values():
            if g is not None and h in g.index.get_level_values(0):
                tot += float(g.xs(h, level=0).sum())
        hosp_sales[h] = tot
    top5 = sorted(hosp_sales, key=lambda h: hosp_sales[h], reverse=True)[:5]

    hosp_rows = []
    for h in top5:
        row = {"医院": h}
        for m, lab in zip(months3, month_labels):
            g = month_net[str(m)]
            if g is not None and h in g.index.get_level_values(0):
                kser = g.xs(h, level=0)
                sales = float(kser.sum())
                valid_keys = kser[kser > 0].index
                n_valid = len(valid_keys)
                fm = first_purchase.reindex(valid_keys).dt.to_period("M")
                n_new = int((fm == m).sum())
                n_old = n_valid - n_new
            else:
                sales, n_valid, n_new, n_old = 0.0, 0, 0, 0
            row[f"{lab} 销量"] = round(sales, 1)
            row[f"{lab} 新患"] = n_new
            row[f"{lab} 老患"] = n_old

        # R12M DOT（回滚12个月，到 target 月末）
        r_start = (target_p - 11).to_timestamp()
        r_end = target_p.to_timestamp(how="end")
        s12 = d[(d["销售时间"] >= r_start) & (d["销售时间"] <= r_end)]
        s12h = s12[s12["医疗单位"] == h]
        net12 = s12h.groupby("key")["销售数量"].sum()
        pos12 = net12[net12 > 0]
        dot = float(pos12.sum()) / pos12.index.nunique() if len(pos12) else 0.0

        # 25年 DOT（2025 全年固定基线）
        y25s = pd.Timestamp(2025, 1, 1)
        y25e = pd.Timestamp(2025, 12, 1).to_period("M").to_timestamp(how="end")
        s25 = d[(d["销售时间"] >= y25s) & (d["销售时间"] <= y25e)]
        s25h = s25[s25["医疗单位"] == h]
        net25 = s25h.groupby("key")["销售数量"].sum()
        pos25 = net25[net25 > 0]
        dot25 = float(pos25.sum()) / pos25.index.nunique() if len(pos25) else 0.0

        yoy = (dot - dot25) / dot25 if dot25 else None
        row["R12M DOT"] = round(dot, 2)
        row["25年DOT"] = round(dot25, 2)
        row["同比"] = round(yoy, 3) if yoy is not None else None
        hosp_rows.append(row)

    hosp_df = pd.DataFrame(hosp_rows)

    # ---- 医生明细（每家 TOP3，近两个月）----
    m_prev, m_cur = months3[1], months3[2]
    lab_prev, lab_cur = month_labels[1], month_labels[2]
    doc_rows = []

    def _doc_metrics(sub, m):
        ms, me = m.to_timestamp(), m.to_timestamp(how="end")
        s = sub[(sub["销售时间"] >= ms) & (sub["销售时间"] <= me)]
        if len(s) == 0:
            return {}
        g = s.groupby("处方医生")["销售数量"].sum()
        net_key = s.groupby(["处方医生", "key"])["销售数量"].sum()
        pat = net_key[net_key > 0].reset_index()["处方医生"].value_counts().to_dict()
        return {doc: {"销量": float(g.get(doc, 0)), "患者数": int(pat.get(doc, 0))}
                for doc in g.index}

    for h in top5:
        sub = d[d["医疗单位"] == h]
        prev_m = _doc_metrics(sub, m_prev)
        cur_m = _doc_metrics(sub, m_cur)
        docs = set(prev_m) | set(cur_m)
        docs.discard("nan"); docs.discard("")
        docs_sorted = sorted(docs, key=lambda x: cur_m.get(x, {}).get("销量", 0), reverse=True)[:3]
        for doc in docs_sorted:
            pv = prev_m.get(doc, {"销量": 0.0, "患者数": 0})
            cv = cur_m.get(doc, {"销量": 0.0, "患者数": 0})
            sv_prev, sv_cur = pv["销量"], cv["销量"]
            pt_prev, pt_cur = pv["患者数"], cv["患者数"]
            sl = (sv_cur - sv_prev) / sv_prev if sv_prev else None
            pl = (pt_cur - pt_prev) / pt_prev if pt_prev else None
            doc_rows.append({
                "医院": h,
                "医生": _mask_name(doc),
                f"{lab_prev}销量": round(sv_prev, 1),
                f"{lab_cur}销量": round(sv_cur, 1),
                "销量环比": round(sl, 3) if sl is not None else None,
                f"{lab_prev}患者数": pt_prev,
                f"{lab_cur}患者数": pt_cur,
                "患者数环比": round(pl, 3) if pl is not None else None,
            })
    doctor_df = pd.DataFrame(doc_rows)
    return hosp_df, doctor_df


# ------------------------- 运营分析 -------------------------
def _clean_ind(raw, mapping):
    """原始适应症 -> 标准值；空 / 缺失 -> 其他。"""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return "其他"
    s = str(raw).strip()
    if s == "" or s.lower() == "nan":
        return "其他"
    return mapping.get(s, "其他") if mapping else "其他"


def _mask_patient(key):
    """患者标识脱敏：返回 P + md5 前 8 位，不暴露 oneid / 姓名 / 电话。"""
    import hashlib
    return "P" + hashlib.md5(str(key).encode("utf-8")).hexdigest()[:8]


def _summary_box(text):
    """在板块底部展示一段自动生成的总结分析文字。"""
    st.info("📌 **自动总结**：" + text)


def indication_global(d_all, pt_all, disp_start, disp_end, effective_mapping, std_qty):
    """各适应症全局对比（截至 disp_end）：累计患者数 / 窗口新患数 / R12M DOT / 规范治疗率。
    患者按全量末次购药适应症(清洗)整体归因。复用 treatment_rate / r12m_dot。"""
    d = d_all.copy()
    if "key" not in d.columns:
        d["key"] = [make_key(o, t, m, n) for o, t, m, n in
                    zip(d["oneid"], d["开票抬头"], d["会员号"], d["会员姓名"])]
        d = d[d["key"].notna()].copy()
    d["销售数量"] = pd.to_numeric(d["销售数量"], errors="coerce").fillna(0)
    ds_p = pd.Timestamp(disp_start).to_period("M")
    de_p = pd.Timestamp(disp_end).to_period("M")

    # 患者级末次适应症归因（全量）
    last = d.sort_values("销售时间").groupby("key").last().reset_index()
    last["标准适应症"] = last["适应症"].map(lambda x: _clean_ind(x, effective_mapping))
    cnt = last.groupby("标准适应症").size()

    # 窗口新患（首购月落在窗口）
    first_m = pt_all.set_index("key")["首次购药时间"].dt.to_period("M")
    in_win = last[last["key"].map(first_m).between(ds_p, de_p)]
    np_cnt = in_win.groupby("标准适应症").size() if len(in_win) else pd.Series(dtype=int)

    # R12M DOT（disp_end 回滚12月）
    dot_trend = r12m_dot(d, pt_all, disp_start, disp_end, effective_mapping)
    if len(dot_trend):
        _m = pd.to_datetime(dot_trend["月份"]).dt.to_period("M")
        dot_last = dot_trend[_m == de_p].set_index("适应症")["DOT"]
    else:
        dot_last = pd.Series(dtype=float)

    # 规范治疗率（disp_end 月）
    tr = treatment_rate(d, pt_all, disp_end, effective_mapping, std_qty)
    tr_rate = tr.set_index("适应症")["达标率"] if len(tr) else pd.Series(dtype=float)

    inds = sorted(set(cnt.index) | set(np_cnt.index) | set(dot_last.index) | set(tr_rate.index))
    rows = [{
        "标准适应症": ind,
        "累计患者数": int(cnt.get(ind, 0)),
        "窗口新患数": int(np_cnt.get(ind, 0)),
        "R12M DOT": round(float(dot_last.get(ind, 0)), 2),
        "规范治疗率": round(float(tr_rate.get(ind, 0)), 3),
    } for ind in inds]
    return pd.DataFrame(rows).sort_values("累计患者数", ascending=False).reset_index(drop=True)


def retention_cohort(df_all, pt_all, cohort_start, cohort_end):
    """按首购月 cohort，计算首购后 +0/+1/+3/+6/+12 月的留存率（有购买即留存）。
    返回 DataFrame：行=首购月，列=各 offset 留存率。"""
    d = df_all.copy()
    d["key"] = [make_key(o, t, m, n) for o, t, m, n in
                zip(d["oneid"], d["开票抬头"], d["会员号"], d["会员姓名"])]
    d = d[d["key"].notna()].copy()
    d["销售数量"] = pd.to_numeric(d["销售数量"], errors="coerce").fillna(0)
    d_pos = d[d["销售数量"] > 0].copy()
    d_pos["月份"] = d_pos["销售时间"].dt.to_period("M")
    monthly = {m: set(s) for m, s in d_pos.groupby("月份")["key"].apply(list).items()}
    max_m = max(monthly.keys()) if monthly else None

    first_m = d_pos["销售时间"].dt.to_period("M").groupby(d_pos["key"]).min()
    cs_p = pd.Timestamp(cohort_start).to_period("M")
    ce_p = pd.Timestamp(cohort_end).to_period("M")
    cohort_patients = first_m[(first_m >= cs_p) & (first_m <= ce_p)]

    offsets = [0, 1, 3, 6, 12]
    recs = []
    if len(cohort_patients):
        for co_m, grp in cohort_patients.groupby(cohort_patients.values):
            keyset = set(grp.index.tolist())
            row = {"首购月": co_m.strftime("%Y-%m"), "首购患者数": len(keyset)}
            for off in offsets:
                t_m = co_m + off
                if max_m is not None and t_m <= max_m and t_m in monthly:
                    row[f"+{off}月留存"] = round(len(keyset & monthly[t_m]) / len(keyset), 3)
                else:
                    row[f"+{off}月留存"] = None
            recs.append(row)
    return pd.DataFrame(recs)


def churn_risk(df_all, pt_all, target_month, n_months=3):
    """断药预警：历史有购买、且末次购买月早于 (target_month - n_months) 的活跃患者清单。
    返回脱敏清单（不暴露 oneid / 姓名 / 电话）。"""
    d = df_all.copy()
    d["key"] = [make_key(o, t, m, n) for o, t, m, n in
                zip(d["oneid"], d["开票抬头"], d["会员号"], d["会员姓名"])]
    d = d[d["key"].notna()].copy()
    d["销售数量"] = pd.to_numeric(d["销售数量"], errors="coerce").fillna(0)
    t_p = pd.Timestamp(target_month).to_period("M")

    d_pos = d[d["销售数量"] > 0]
    last_m = d_pos["销售时间"].dt.to_period("M").groupby(d_pos["key"]).max()
    cum = d.groupby("key")["销售数量"].sum()
    first_m = pt_all.set_index("key")["首次购药时间"].dt.to_period("M")
    last_row = d.sort_values("销售时间").groupby("key").last()
    last_row["适应症"] = last_row["适应症"].fillna("（缺失）")
    last_row["药房名称"] = last_row["药房名称"].fillna("（缺失）")

    cut = t_p - n_months
    risk_keys = last_m[last_m < cut].index
    recs = []
    for k in risk_keys:
        lm = last_m.get(k)
        if pd.isna(lm):
            continue
        fm = first_m.get(k)
        recs.append({
            "患者标识(脱敏)": _mask_patient(k),
            "首购月": fm.strftime("%Y-%m") if pd.notna(fm) else "",
            "末次购药月": lm.strftime("%Y-%m"),
            "断药月数": int((t_p.year - lm.year) * 12 + (t_p.month - lm.month)),
            "累计盒数": int(cum.get(k, 0)),
            "末次适应症": last_row.loc[k, "适应症"] if k in last_row.index else "",
            "末次药房": last_row.loc[k, "药房名称"] if k in last_row.index else "",
        })
    df_risk = pd.DataFrame(recs)
    if len(df_risk):
        df_risk = df_risk.sort_values("断药月数", ascending=False).reset_index(drop=True)
    return df_risk


# ------------------------- CSV 下载辅助 -------------------------
def _download_csv(df, filename, label, key):
    """把 DataFrame 导出为 UTF-8-SIG 的 CSV（Excel 打开中文不乱码），提供下载按钮。"""
    if df is None or len(df) == 0:
        st.caption("（无数据可下载）")
        return
    csv = df.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(label=label, data=csv, file_name=filename,
                       mime="text/csv", key=key)


def _xlsx_val(v):
    """把 pandas 的 NaN / NaT / None 转成 openpyxl 可写的空值；其余原样返回。"""
    if v is None:
        return None
    if isinstance(v, float) and v != v:  # NaN
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _write_excel_bytes(sheets):
    """sheets: list of (DataFrame, sheet_name)。
    直接用 openpyxl 逐行写出，规避 pandas3 + openpyxl3.1 的
    IndexError: At least one sheet must be visible 的版本组合 bug。
    空表/无列表的 DataFrame 自动替换为提示占位表，保证至少一个可见 sheet。"""
    from openpyxl import Workbook
    wb = Workbook()
    used = {}
    first = True
    for df, raw_name in sheets:
        if df is None or (hasattr(df, "empty") and (df.empty or len(df.columns) == 0)):
            df = pd.DataFrame({"(提示)": ["该表在所选范围内无数据"]})
        name = str(raw_name)[:31]
        if name in used:
            used[name] += 1
            name = f"{name[:27]}_{used[name]}"
        else:
            used[name] = 0
        ws = wb.active if first else wb.create_sheet()
        first = False
        ws.title = name
        ws.append([str(c) for c in df.columns])
        for _, row in df.iterrows():
            ws.append([_xlsx_val(v) for v in row.tolist()])
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf.getvalue()


def _fmt_month(df):
    """若有 月份 列则转成 'YYYY-MM' 字符串；空表/无列返回提示占位表。"""
    if df is None or (hasattr(df, "empty") and (df.empty or len(df.columns) == 0)):
        return pd.DataFrame({"(提示)": ["该表在所选范围内无数据"]})
    d = df.copy()
    if "月份" in d.columns:
        d["月份"] = pd.to_datetime(d["月份"]).dt.strftime("%Y-%m")
    return d


def _download_all_xlsx(pool, pharm_df, ind_df, unmapped, web_map_df, kpi_dict,
                       filename, label, key, extra_sheets=None):
    """把看板所有内容打包成一个 Excel（多 sheet，UTF-8），一次下载全部。
    extra_sheets: list of (DataFrame, sheet_name)，附加到末尾。"""
    sheets = []
    summ = pd.DataFrame([{"指标": k, "值": v} for k, v in kpi_dict.items()])
    sheets.append((summ, "看板概要"))
    # 患者池累计（月份转 年-月 字符串）
    pool_out = pool.copy()
    if "月份" in pool_out.columns:
        pool_out["月份"] = pd.to_datetime(pool_out["月份"]).dt.strftime("%Y-%m")
    sheets.append((pool_out, "患者池累计"))
    # 药房分布
    if pharm_df is not None and len(pharm_df) and len(pharm_df.columns):
        sheets.append((pharm_df, "药房分布"))
    # 适应症分布
    if ind_df is not None and len(ind_df) and len(ind_df.columns):
        sheets.append((ind_df, "适应症分布"))
    # 未映射明细
    if unmapped is not None and len(unmapped) and len(unmapped.columns):
        sheets.append((unmapped, "未映射明细"))
    # 网页新增映射
    if web_map_df is not None and len(web_map_df) and len(web_map_df.columns):
        sheets.append((web_map_df, "网页新增映射"))
    # 附加 sheet（趋势分析、药房维度等）
    if extra_sheets:
        for _df, _name in extra_sheets:
            if _df is not None and len(_df) and len(_df.columns):
                _df2 = _df.copy()
                if "月份" in _df2.columns:
                    _df2["月份"] = pd.to_datetime(_df2["月份"]).dt.strftime("%Y-%m")
                sheets.append((_df2, _name))
    data = _write_excel_bytes(sheets)
    st.download_button(
        label=label, data=data, file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=key,
    )


# ------------------------- 主程序 -------------------------
def main():
    st.set_page_config(page_title="瑞视 · 乌帕替尼 DTP 运营看板", layout="wide")
    st.title("瑞视 · 乌帕替尼(瑞福) DTP 销售运营看板")
    st.caption("RuiLens · 艾伯维乌帕替尼(瑞福) DTP 药房销售与运营分析平台 ｜ v2.9")

    # ---- 时间筛选默认值（按钮通过 session_state 回填日期框）----
    _today = date.today()
    _defaults = {
        "user_mapping": {},  # 网页上用户新增的原始适应症->标准值映射
    }
    for _k, _v in _defaults.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    def _shift_month(d, n):
        return (pd.Timestamp(d).to_period("M") + n).to_timestamp().date()

    # ---- 侧边栏：数据源 ----
    st.sidebar.header("① 数据源")
    st.sidebar.caption("部署版：请上传销售明细表（含患者隐私，不随仓库发布）")
    uploaded = st.sidebar.file_uploader("上传销售明细表 (.xlsx)", type=["xlsx"])
    strip_pii = st.sidebar.checkbox(
        "🛡 上传前自动剔除隐私列",
        value=True,
        help="默认开启：上传即把 电话 / 姓名 / 会员号 等直接标识符置空，"
             "计算结果不受影响，但敏感信息不会进入服务器内存。",
    )

    # ---- 侧边栏：时间筛选（按钮化快捷选择 + 年/月下拉，统一到"年月"）----
    st.sidebar.header("② 时间筛选")

    _yrs = list(range(2022, _today.year + 2))
    _mos = list(range(1, 13))

    def _ym_selector(label, ykey, mkey, default_date):
        """年 + 月 两个下拉，返回 date(year, month, 1)。快捷按钮通过 session_state 回填。"""
        if ykey not in st.session_state:
            st.session_state[ykey] = default_date.year
        if mkey not in st.session_state:
            st.session_state[mkey] = default_date.month
        col_y, col_m = st.sidebar.columns(2)
        with col_y:
            y = st.selectbox("年", _yrs, index=_yrs.index(st.session_state[ykey]), key=ykey)
        with col_m:
            m = st.selectbox("月", _mos, index=_mos.index(st.session_state[mkey]), key=mkey)
        return date(y, m, 1)

    # YTD 开始年月
    st.sidebar.markdown("**YTD 开始年月**")
    ytd_start = _ym_selector("YTD 开始年月", "ytd_y", "ytd_m", date(2024, 1, 1))

    # 展示区间（起 ~ 止月）
    st.sidebar.markdown("**展示区间（起 ~ 止月）**")
    disp_start = _ym_selector("展示起始月", "ds_y", "ds_m", date(2025, 5, 1))
    disp_end = _ym_selector("展示结束月", "de_y", "de_m", date(2026, 4, 1))

    # 分布图目标年月
    st.sidebar.markdown("**分布图目标年月**")
    st.sidebar.caption("分布图按所选年月统计")
    target_month = _ym_selector("选择目标年月", "t_y", "t_m", date(2026, 4, 1))
    st.sidebar.info(f"当前选择：{target_month.strftime('%Y年%m月')}")

    # ---- 侧边栏：适应症映射 ----
    st.sidebar.header("③ 适应症清洗映射")
    st.sidebar.caption("默认使用仓库内置映射表，如需覆盖可上传")
    map_file = st.sidebar.file_uploader("上传映射表 (.xlsx，可选)", type=["xlsx"])
    if map_file is not None:
        mapping, mp_df = load_mapping(uploaded_file=map_file)
    else:
        mapping, mp_df = load_mapping(path=DEFAULT_MAP_PATH)

    if uploaded is None:
        st.info("👆 请在左侧上传销售明细表 (.xlsx) 后再查看看板。")
        return
    df = load_data(uploaded, None)

    # 隐私保护：剔除直接标识符列（默认开启）
    if strip_pii:
        df, _dropped = strip_pii_columns(df)
        if _dropped:
            st.sidebar.success(f"🛡 已剔除隐私列：{', '.join(_dropped)}")

    # 全量数据（不受侧边栏药房筛选影响），供「单药房维度分析」Tab 使用
    df_all = df.copy()

    # ---- 侧边栏：药房筛选（可多选）----
    st.sidebar.header("④ 药房筛选（可选）")
    st.sidebar.caption("按药房名称筛选；留空表示不筛选（展示全部有数据药房）。")

    def _pharm_opts():
        if "药房名称" not in df.columns:
            return []
        s = df["药房名称"].dropna().astype(str).str.strip()
        s = s[s != ""]
        return sorted(s.unique().tolist())

    sel_pharm = st.sidebar.multiselect("药房名称", _pharm_opts(), key="flt_pharm")
    if sel_pharm:
        df = df[df["药房名称"].astype(str).str.strip().isin(sel_pharm)]

    if len(df) == 0:
        st.warning("当前药房筛选条件下无数据，请调整或清空「④ 药房筛选」。")
        return

    # 合并「网页新增映射」与文件映射，得到生效映射
    _base = mapping if mapping is not None else {}
    _user = st.session_state.get("user_mapping", {})
    effective_mapping = {**_base, **_user}
    effective_mapping = effective_mapping if effective_mapping else None

    with st.spinner("计算患者维度…"):
        d_keyed, pt = build_patient_table(df)
        _, pt_all = build_patient_table(df_all)  # 全局患者表：单药房Tab的新/老患按全局首次购药判定
        pool = patient_pool(d_keyed, pt, ytd_start, disp_start, disp_end)
        pharm = pharmacy_dist(d_keyed, target_month)
        ind_counts, unmapped = indication_dist(d_keyed, pt, target_month, effective_mapping)
        std_qty = std_qty_from_map(mp_df)
        sales = sales_trend(d_keyed, pt, disp_start, disp_end)
        np_ind = np_by_indication(d_keyed, pt, disp_start, disp_end, effective_mapping)
        treat = treatment_rate(d_keyed, pt, target_month, effective_mapping, std_qty)
        dot = r12m_dot(d_keyed, pt, disp_start, disp_end, effective_mapping)
        pharm_metrics = pharmacy_table(d_keyed, pt, ytd_start, target_month)
        # 药房维度只保留有数据的药房：关键计数项任一非 0 即视为有数据
        _pharm_has_data = (
            (pharm_metrics["累计OP数量"] > 0)
            | (pharm_metrics["目标月活跃人数"] > 0)
            | (pharm_metrics["应购患者数"] > 0)
            | (pharm_metrics["R12M DOT"] > 0)
        )
        pharm_metrics = pharm_metrics[_pharm_has_data].reset_index(drop=True)
        # 临床分析 TOP5 医院（按医疗单位，使用全量数据 df_all / 全局患者口径 pt_all）
        hosp_df, doctor_df = clinical_top5(df_all, pt_all, target_month, effective_mapping)
        # 运营分析（适应症全局对比 / 留存 cohort / 断药预警），均基于全量数据 df_all
        ind_sum = indication_global(df_all, pt_all, disp_start, disp_end, effective_mapping, std_qty)
        cohort = retention_cohort(df_all, pt_all, disp_start, target_month)
        churn = churn_risk(df_all, pt_all, target_month, 3)

    # 分布图里复用的 DataFrame（也用于打包下载）
    pharm_df = pharm.rename_axis("药房名称").reset_index(name="患者数")
    ind_df = ind_counts.rename_axis("标准适应症").reset_index(name="患者数")
    web_map_df = pd.DataFrame(
        [{"原始适应症": k, "标准值": v} for k, v in st.session_state.get("user_mapping", {}).items()]
    )

    # ---- 顶部 KPI（常驻）----
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("累计患者数", int(pool["累计患者"].iloc[-1]) if len(pool) else 0)
    kpi2.metric("门店数（全量）", df["药房名称"].nunique())
    kpi3.metric("覆盖城市数（全量）", df["药房名称"].astype(str).str[:2].nunique())

    # ---- 一键下载全部看板数据（常驻）----
    st.markdown("#### 📦 一键下载全部看板数据")
    st.caption("把当前筛选条件下的所有看板内容打包成一个 Excel（多 sheet），一次下载。")
    _kpi = {
        "累计患者数": int(pool["累计患者"].iloc[-1]) if len(pool) else 0,
        "门店数（全量）": int(df["药房名称"].nunique()),
        "覆盖城市数（全量）": int(df["药房名称"].astype(str).str[:2].nunique()),
        "YTD开始年月": ytd_start.strftime("%Y-%m"),
        "展示起始月": disp_start.strftime("%Y-%m"),
        "展示结束月": disp_end.strftime("%Y-%m"),
        "分布图目标月": target_month.strftime("%Y-%m"),
        "数据来源": "页面上传",
        "映射表": ("页面上传" if map_file is not None else "仓库内置"),
        "生成时间": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    # 药房维度：导出前把比例格式化为字符串，便于 Excel 直接阅读
    _pharm_xlsx = pharm_metrics.copy()
    _pharm_xlsx["目标月复购率"] = _pharm_xlsx["目标月复购率"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "")
    _pharm_xlsx["复购率环比"] = _pharm_xlsx["复购率环比"].map(lambda x: f"{x:.1%}" if pd.notna(x) else "")
    _pharm_xlsx["R12M DOT"] = _pharm_xlsx["R12M DOT"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    _pharm_xlsx["R12M DOT环比"] = _pharm_xlsx["R12M DOT环比"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    _download_all_xlsx(
        pool, pharm_df, ind_df, unmapped, web_map_df, _kpi,
        "患者池看板_全部数据.xlsx",
        "⬇ 下载全部看板数据（Excel · 含患者池/药房/适应症/趋势等）",
        "dl_all",
        extra_sheets=[
            (sales.assign(月份=sales["月份"].dt.strftime("%Y-%m")), "销量趋势"),
            (np_ind.assign(月份=np_ind["月份"].dt.strftime("%Y-%m")), "NP数量"),
            (treat.assign(达标率=treat["达标率"].map(lambda x: f"{x:.1%}")), "规范治疗率"),
            (dot.assign(月份=dot["月份"].dt.strftime("%Y-%m")), "R12M-DOT"),
            (_pharm_xlsx, "药房维度"),
            (hosp_df, "临床_TOP5医院"),
            (doctor_df, "临床_医生明细"),
            (ind_sum, "运营_适应症对比"),
            (cohort, "运营_留存漏斗"),
            (churn, "运营_断药预警"),
        ],
    )

    # ============ 看板 Tab 切换 ============
    tab_pool, tab_pharm, tab_trend, tab_single, tab_clinical, tab_ops = st.tabs(
        ["🏥 患者池与分布", "🏪 药房维度", "📈 趋势分析", "🔬 单药房维度分析",
         "🏨 临床分析-TOP5医院", "🛠 运营分析"])

    with tab_pool:
        # ---- 患者池累计 堆叠柱状图 ----
        st.subheader("患者池累计情况")
        st.caption("累计患者 = 自 YTD 开始日起窗口内有购药记录的患者滚动累积（首购早但期内复购者从复购月起入池）；YTD老患 = 累计患者 − 当月新患；当月新患 = 全局首购月 == 当月。柱状图逐月累积。")
        if len(pool):
            fig = go.Figure()
            fig.add_bar(x=pool["月份"], y=pool["YTD老患"], name="YTD老患", marker_color="#1f4e79",
                        text=pool["YTD老患"], textposition="inside", insidetextanchor="middle")
            fig.add_bar(x=pool["月份"], y=pool["当月新患"], name="当月新患", marker_color="#9dc3e6",
                        text=pool["当月新患"], textposition="outside")
            fig.update_layout(barmode="stack", height=460,
                              xaxis_tickformat="%Y-%m", yaxis_title="患者数")
            fig.update_traces(texttemplate="%{text}", textfont_size=10)
            st.plotly_chart(fig, width="stretch")
            # 下载：患者池累计（月份/活跃/新患/复购/老患/累计）
            pool_dl = pool.copy()
            pool_dl["月份"] = pool_dl["月份"].dt.strftime("%Y-%m")
            _download_csv(pool_dl, "患者池累计.csv",
                          "⬇ 下载患者池累计数据（月份/活跃/新患/复购/老患/累计）", "dl_pool")

            # ---- 整体患者趋势：累计线 + 当月活跃堆叠柱 ----
            st.subheader("整体患者趋势")
            st.caption("累计患者（红线）= 自 YTD 起滚动累积池（含断药患者，持续走高）；"
                       "当月活跃患者（柱）= 当月实际有购药的患者，即真实月度销售基数，"
                       "由【当月新患 + 当月复购老患】堆叠。柱高反映真实当月业务，线反映盘子规模。")
            ft = go.Figure()
            ft.add_bar(x=pool["月份"], y=pool["当月新患"], name="当月新患",
                       marker_color="#9dc3e6", text=pool["当月新患"],
                       textposition="outside")
            ft.add_bar(x=pool["月份"], y=pool["当月老患(复购)"], name="当月复购老患",
                       marker_color="#1f4e79", text=pool["当月老患(复购)"],
                       textposition="inside", insidetextanchor="middle")
            ft.add_trace(go.Scatter(
                x=pool["月份"], y=pool["累计患者"], name="累计患者",
                mode="lines+markers", line=dict(color="#c00000", width=3)))
            ft.update_layout(
                barmode="stack", height=460, xaxis_tickformat="%Y-%m",
                yaxis_title="患者数",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
            ft.update_traces(texttemplate="%{text}", textfont_size=10)
            st.plotly_chart(ft, width="stretch")
        else:
            st.warning("当前时间筛选无数据，请调整 YTD 开始日或展示区间。")

        # ---- 分布图 ----
        _date_label = target_month.strftime("%Y年%m月")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader(f"{_date_label} 药房患者分布")
            if len(pharm):
                figp = px.pie(names=pharm.index, values=pharm.values,
                              hole=0.4, title="按末次购药药房")
                figp.update_traces(textinfo="percent+label")
                st.plotly_chart(figp, width="stretch")
                _download_csv(pharm_df, "药房分布.csv", "⬇ 下载药房分布数据", "dl_pharm")
            else:
                st.info("该月无销售记录。")
        with c2:
            st.subheader(f"{_date_label} 适应症分布")
            if len(ind_counts):
                figi = px.pie(names=ind_counts.index, values=ind_counts.values,
                              hole=0.4, title="按末次购药适应症（清洗后）")
                figi.update_traces(textinfo="percent+label")
                st.plotly_chart(figi, width="stretch")
                _download_csv(ind_df, "适应症分布.csv", "⬇ 下载适应症分布数据", "dl_ind")
            else:
                st.info("该月无销售记录。")

        # ---- 未映射明细（可在网页直接补填映射）----
        st.subheader("适应症清洗维护：未映射明细")
        st.caption("下表为当月患者中、映射表里找不到的原始适应症。可在右侧「标准值」列填写（如 RA / AD / 其他），"
                   "点「应用网页新增映射」即时生效；空值适应症已自动归为「其他」。")
        if len(unmapped):
            edit_df = unmapped.copy()
            edit_df["标准值（填写后点应用）"] = ""
            edited = st.data_editor(
                edit_df,
                num_rows="dynamic",
                disabled=["原始适应症", "患者数"],
                key="unmap_editor",
                width="stretch",
            )
            if st.button("应用网页新增映射", key="apply_map"):
                added = 0
                for _, row in edited.iterrows():
                    raw = str(row["原始适应症"]).strip()
                    clean = str(row["标准值（填写后点应用）"]).strip()
                    if raw and clean and clean.lower() != "nan":
                        st.session_state.user_mapping[raw] = clean
                        added += 1
                if added:
                    st.toast(f"已新增 {added} 条网页映射，重新计算中…")
                    st.rerun()
                else:
                    st.info("没有填写任何标准值，未做改动。")
            # 未映射明细可下载（供维护回写）
            _download_csv(unmapped, "未映射明细.csv", "⬇ 下载未映射明细", "dl_unmap")
        else:
            st.success("当月所有适应症均已映射到标准值 ✅（空值已自动归为「其他」）")

        # ---- 映射表查看 ----
        if mp_df is not None:
            with st.expander("查看已上传的映射表"):
                # 规范治疗数量列含数字与"其他"混合，转字符串避免 pyarrow 类型推断报错
                st.dataframe(mp_df.astype(str), width="stretch")
        if st.session_state.get("user_mapping"):
            with st.expander("查看网页新增的映射", expanded=True):
                _um = pd.DataFrame(
                    [{"原始适应症": k, "标准值": v} for k, v in st.session_state.user_mapping.items()]
                )
                st.dataframe(_um.astype(str), width="stretch")
                _download_csv(_um, "网页新增映射.csv", "⬇ 下载网页新增映射", "dl_um")
                if st.button("清空网页新增映射", key="clear_map"):
                    st.session_state.user_mapping = {}
                    st.toast("已清空网页新增映射")
                    st.rerun()

        # ---- 患者池自动总结 ----
        if len(pool):
            _last = pool.iloc[-1]
            _top_pharm = pharm.sort_values(ascending=False).head(1)
            _top_ind = ind_counts.sort_values(ascending=False).head(1)
            _summary_box(
                f"截至 {_last['月份'].strftime('%Y-%m')}，累计患者 {int(_last['累计患者'])} 人"
                f"（当月活跃 {int(_last['当月活跃患者'])}、新患 {int(_last['当月新患'])}、"
                f"复购老患 {int(_last['当月老患(复购)'])}）。"
                f"累计线持续走高（含历史断药患者），活跃柱反映真实当月业务基数；"
                f"当月药房分布 Top1：{_top_pharm.index[0]}（{int(_top_pharm.iloc[0])} 人）；"
                f"适应症分布 Top1：{_top_ind.index[0]}（{int(_top_ind.iloc[0])} 人）。"
            )

    with tab_pharm:
        # ============ 药房维度 ============
        st.subheader(f"{target_month.strftime('%Y年%m月')} 药房维度汇总")
        st.caption(
            "患者按每笔销售记录归属药房，可在多个药房出现；累计OP = YTD 累计老患者且在该药房有过购买；"
            "复购率 = 应购患者中目标月实际复购占比（应购 = 末次正销量购药 + 盒数×30 天落在目标月内）；"
            "环比 = 本月值 − 上月值。"
        )
        if len(pharm_metrics):
            # 展示用短列名
            display = pharm_metrics.rename(columns={
                "目标月活跃人数": "活跃人数",
                "目标月OP人数": "OP人数",
                "目标月NP人数": "NP人数",
                "目标月复购率": "复购率",
            }).copy()
            st.dataframe(
                display,
                column_config={
                    "复购率": st.column_config.NumberColumn(format="%.1%"),
                    "复购率环比": st.column_config.NumberColumn(format="%.1%"),
                    "R12M DOT": st.column_config.NumberColumn(format="%.2f"),
                    "R12M DOT环比": st.column_config.NumberColumn(format="%.2f"),
                },
                use_container_width=True,
                hide_index=True,
            )
            _download_csv(pharm_metrics, "药房维度.csv", "⬇ 下载药房维度数据", "dl_pharm_metrics")
        else:
            st.info("当前目标月无数据。")

        # ---- 药房维度自动总结 ----
        if len(pharm_metrics):
            _top_act = pharm_metrics.loc[pharm_metrics["目标月活跃人数"].idxmax()]
            _top_op = pharm_metrics.loc[pharm_metrics["累计OP数量"].idxmax()]
            _top_dot = pharm_metrics.loc[pharm_metrics["R12M DOT"].idxmax()]
            _summary_box(
                f"目标月活跃人数最高：{_top_act['药房名称']}（{int(_top_act['目标月活跃人数'])} 人）；"
                f"累计OP最多：{_top_op['药房名称']}（{int(_top_op['累计OP数量'])} 人）；"
                f"R12M DOT 最高：{_top_dot['药房名称']}（{_top_dot['R12M DOT']:.2f}）。"
            )

    with tab_trend:
        # ============ 趋势分析 ============
        # ---- 销量趋势：每月新老患者销量（堆叠柱）----
        st.subheader("销量趋势（每月新老患者销量，盒）")
        st.caption("老患销量 = 当月有购药且首次购药早于当月的患者，其当月净销量总和；"
                   "新患销量 = 首次购药落在该月的患者当月净销量。均为退货正负抵消后的净销量。")
        if len(sales):
            fig_s = go.Figure()
            fig_s.add_bar(x=sales["月份"], y=sales["老患销量"], name="老患销量", marker_color="#1f4e79",
                          text=sales["老患销量"], textposition="outside")
            fig_s.add_bar(x=sales["月份"], y=sales["新患销量"], name="新患销量", marker_color="#9dc3e6",
                          text=sales["新患销量"], textposition="outside")
            fig_s.update_layout(barmode="stack", height=420, xaxis_tickformat="%Y-%m", yaxis_title="净销量（盒）")
            fig_s.update_traces(texttemplate="%{text:.0f}", textfont_size=9)
            st.plotly_chart(fig_s, width="stretch")
            _download_csv(sales.assign(月份=sales["月份"].dt.strftime("%Y-%m")),
                          "销量趋势.csv", "⬇ 下载销量趋势", "dl_sales")
        else:
            st.info("当前时间筛选无数据。")

        # ---- NP 数量：分适应症当月新患（折线）----
        st.subheader("NP 数量（分适应症当月新患数）")
        st.caption("分适应症折线；仅展示窗口内首次购药落在该月的新患。")
        if len(np_ind):
            pivot = np_ind.pivot(index="月份", columns="适应症", values="新患数").fillna(0).reset_index()
            pivot["月份"] = pd.to_datetime(pivot["月份"]).dt.strftime("%Y-%m")
            fig_np = px.line(pivot, x="月份", y=[c for c in pivot.columns if c != "月份"],
                             markers=True, title="当月新患数（按末次适应症）")
            fig_np.update_layout(yaxis_title="新患数", xaxis_title="月份")
            st.plotly_chart(fig_np, width="stretch")
            _download_csv(np_ind.assign(月份=np_ind["月份"].dt.strftime("%Y-%m")),
                          "NP数量.csv", "⬇ 下载NP数量", "dl_np")
        else:
            st.info("当前区间无新患。")

        # ---- 规范治疗率 ----
        st.subheader(f"{target_month.strftime('%Y年%m月')} 规范治疗率")
        st.caption("去除当月新患，老患者从首次购药到当月末累计净购药盒数 ≥ 规范治疗数量 的占比；"
                   "「其他」/未映射适应症无规范治疗量，不计入。")
        if len(treat):
            fig_tr = px.bar(treat, x="适应症", y="达标率", text="达标率", title="各适应症规范治疗率")
            fig_tr.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            fig_tr.update_layout(yaxis_title="达标率", yaxis_tickformat=".0%")
            st.plotly_chart(fig_tr, width="stretch")
            _treat_dl = treat.assign(达标率=treat["达标率"].map(lambda x: f"{x:.1%}"))
            _download_csv(_treat_dl, "规范治疗率.csv", "⬇ 下载规范治疗率", "dl_treat")
            with st.expander("查看明细（老患者数 / 达标患者数 / 累计盒数判定）"):
                st.dataframe(_treat_dl, width="stretch")
        else:
            st.info("当前目标月无老患者，或老患者均属「其他/未映射」适应症。")

        # ---- R12M DOT 趋势 ----
        st.subheader("R12M DOT 趋势（分适应症）")
        st.caption("每月向前回滚 12 个月：DOT = 滚动12月净销量盒数总和 / 滚动12月去重患者数。")
        if len(dot):
            pivot = dot.pivot(index="月份", columns="适应症", values="DOT").fillna(0).reset_index()
            pivot["月份"] = pd.to_datetime(pivot["月份"]).dt.strftime("%Y-%m")
            fig_dot = px.line(pivot, x="月份", y=[c for c in pivot.columns if c != "月份"],
                              markers=True, title="R12M DOT（盒/患者）")
            fig_dot.update_layout(yaxis_title="DOT（盒/患者）", xaxis_title="月份")
            st.plotly_chart(fig_dot, width="stretch")
            _download_csv(dot.assign(月份=dot["月份"].dt.strftime("%Y-%m")),
                          "R12M_DOT.csv", "⬇ 下载R12M DOT", "dl_dot")
        else:
            st.info("当前区间无数据。")

        # ---- 趋势分析自动总结 ----
        if len(sales):
            _tot = int(sales["老患销量"].sum() + sales["新患销量"].sum())
            _first = sales.iloc[0]; _lastm = sales.iloc[-1]
            _top_np = np_ind.groupby("适应症")["新患数"].sum().sort_values(ascending=False)
            _txt = (f"展示区间内净销量合计 {_tot} 盒；最新月 {_lastm['月份'].strftime('%Y-%m')} 净销量 "
                    f"{int(_lastm['老患销量'] + _lastm['新患销量'])} 盒"
                    f"（首月 {int(_first['老患销量'] + _first['新患销量'])} 盒）。")
            if len(_top_np):
                _txt += f"新患累计最多适应症：{_top_np.index[0]}（{int(_top_np.iloc[0])} 人）。"
            _summary_box(_txt)

    with tab_single:
        # ============ 单药房维度分析 ============
        st.subheader("🔬 单药房维度分析")
        st.caption(
            "按所选药房展示月度适应症分布与新老患净销量分布。"
            "适应症分布：窗口内每月在该药房有购买的患者，按各自末次购药适应症（清洗后）堆叠；"
            "新老患净销量：每月净销量（退货正负抵消）按患者「全局首次购药月」拆分"
            "（新患 = 首购落在该月，老患 = 首购早于该月）。"
        )

        # 药房下拉（来自全量数据，不受侧边栏药房筛选影响）
        _all_pharms = sorted(df_all["药房名称"].dropna().astype(str).str.strip().unique().tolist())
        if not _all_pharms:
            st.warning("无药房数据。")
        else:
            sel_single = st.selectbox("选择药房", _all_pharms, key="sp_pharm",
                                      index=0 if "成都西三段药房(连锁）" not in _all_pharms
                                      else _all_pharms.index("成都西三段药房(连锁）"))

            # 时间范围（起 ~ 止月）
            sc1, sc2 = st.columns(2)
            with sc1:
                sp_start = _ym_selector("起始月", "sp_sy", "sp_sm", date(2025, 5, 1))
            with sc2:
                sp_end = _ym_selector("结束月", "sp_ey", "sp_em", date(2026, 4, 1))

            # 构造该药房全量（带患者 key），供本 Tab 所有计算复用
            d_pharm = df_all[df_all["药房名称"].astype(str).str.strip() == sel_single].copy()
            d_pharm["key"] = [
                make_key(o, t, m, n) for o, t, m, n in zip(
                    d_pharm["oneid"], d_pharm["开票抬头"],
                    d_pharm["会员号"], d_pharm["会员姓名"])
            ]
            d_pharm = d_pharm[d_pharm["key"].notna()].copy()
            d_pharm["销售数量"] = pd.to_numeric(d_pharm["销售数量"], errors="coerce").fillna(0)

            ind_df, sales_df = single_pharmacy_monthly(
                d_pharm, pt_all, sel_single, sp_start, sp_end, effective_mapping)
            np_df2 = single_pharmacy_np(d_pharm, pt_all, sp_start, sp_end, effective_mapping)
            dot_df2 = r12m_dot(d_pharm, pt_all, sp_start, sp_end, effective_mapping)
            rate_df2 = single_pharmacy_treatment_rate(
                d_pharm, pt_all, sp_start, sp_end, effective_mapping, std_qty)

            # ---- 图1：适应症分布（堆叠柱）----
            st.subheader(f"{sel_single} · 适应症分布（每月患者数）")
            if len(ind_df):
                pivot_ind = ind_df.pivot(index="月份", columns="适应症", values="患者数").fillna(0)
                _cols = [c for c in _IND_ORDER if c in pivot_ind.columns] + \
                        [c for c in pivot_ind.columns if c not in _IND_ORDER]
                fig_ind = px.bar(
                    pivot_ind[_cols].reset_index(), x="月份", y=_cols,
                    title="适应症分布（每月患者数，按末次购药适应症）", barmode="stack",
                )
                fig_ind.update_layout(yaxis_title="患者数", xaxis_tickformat="%Y-%m", height=420)
                st.plotly_chart(fig_ind, width="stretch")
                _download_csv(ind_df.assign(月份=ind_df["月份"].dt.strftime("%Y-%m")),
                              "单药房_适应症分布.csv", "⬇ 下载适应症分布数据", "dl_sp_ind")
            else:
                st.info("该药房在所选时间范围内无销售记录。")

            # ---- 图2：新老患净销量分布（堆叠柱）----
            st.subheader(f"{sel_single} · 新老患净销量分布（每月盒数）")
            if len(sales_df):
                fig_sp = go.Figure()
                fig_sp.add_bar(x=sales_df["月份"], y=sales_df["老患销量"], name="老患销量",
                               marker_color="#1f4e79", text=sales_df["老患销量"], textposition="outside")
                fig_sp.add_bar(x=sales_df["月份"], y=sales_df["新患销量"], name="新患销量",
                               marker_color="#9dc3e6", text=sales_df["新患销量"], textposition="outside")
                fig_sp.update_layout(barmode="stack", height=420,
                                     xaxis_tickformat="%Y-%m", yaxis_title="净销量（盒）")
                fig_sp.update_traces(texttemplate="%{text:.0f}", textfont_size=9)
                st.plotly_chart(fig_sp, width="stretch")
                _download_csv(sales_df.assign(月份=sales_df["月份"].dt.strftime("%Y-%m")),
                              "单药房_新老患销量.csv", "⬇ 下载新老患销量数据", "dl_sp_sales")
            else:
                st.info("该药房在所选时间范围内无销量数据。")

            # ============ 第二屏：深度趋势 ============
            st.divider()
            st.subheader(f"{sel_single} · 深度趋势（R12M DOT / 新患 / 规范治疗率）")

            # 图3：R12M DOT 趋势（按末次适应症）
            st.markdown("**R12M DOT 趋势（按末次购药适应症拆分）**")
            if len(dot_df2):
                fig_dot = px.line(
                    dot_df2, x="月份", y="DOT", color="适应症",
                    title="R12M DOT 趋势（单药房·末次适应症）",
                    category_orders={"适应症": _IND_ORDER},
                )
                fig_dot.update_layout(xaxis_tickformat="%Y-%m", height=420,
                                      yaxis_title="DOT（盒/人·12月）")
                st.plotly_chart(fig_dot, width="stretch")
                _download_csv(dot_df2.assign(月份=dot_df2["月份"].dt.strftime("%Y-%m")),
                              "单药房_R12M_DOT趋势.csv", "⬇ 下载 R12M DOT 趋势", "dl_sp_dot")
            else:
                st.info("该药房在所选时间范围内无数据。")

            # 图4：新患趋势（按末次适应症，所有适应症）
            st.markdown("**新患趋势（按末次购药适应症，全局新患）**")
            if len(np_df2):
                fig_np = px.line(
                    np_df2, x="月份", y="新患数", color="适应症",
                    title="新患趋势（单药房·末次适应症）",
                    category_orders={"适应症": _IND_ORDER},
                )
                fig_np.update_layout(xaxis_tickformat="%Y-%m", height=420,
                                     yaxis_title="新患数")
                st.plotly_chart(fig_np, width="stretch")
                _download_csv(np_df2.assign(月份=np_df2["月份"].dt.strftime("%Y-%m")),
                              "单药房_新患趋势.csv", "⬇ 下载新患趋势", "dl_sp_np")
            else:
                st.info("该药房在所选时间范围内无新患数据。")

            # 图5：规范治疗率（按适应症，subplots：达标/未达标人数柱 + 达标率折线）
            st.markdown("**规范治疗率（按适应症：达标/未达标人数 + 达标率）**")
            if len(rate_df2):
                _rate_inds = [c for c in _IND_ORDER if c in rate_df2["适应症"].unique()]
                if _rate_inds:
                    from plotly.subplots import make_subplots
                    fig_rate = make_subplots(
                        rows=len(_rate_inds), cols=1, shared_xaxes=True,
                        subplot_titles=[f"{i} 规范治疗率" for i in _rate_inds],
                        vertical_spacing=0.06,
                        specs=[[{"secondary_y": True}] for _ in _rate_inds],
                    )
                    for i, ind in enumerate(_rate_inds, start=1):
                        s = rate_df2[rate_df2["适应症"] == ind]
                        fig_rate.add_bar(x=s["月份"], y=s["达标患者数"], name="达标",
                                         marker_color="#2e7d32", row=i, col=1,
                                         legendgroup=ind, showlegend=(i == 1))
                        fig_rate.add_bar(x=s["月份"], y=s["未达标患者数"], name="未达标",
                                         marker_color="#c62828", row=i, col=1,
                                         legendgroup=ind, showlegend=False)
                        fig_rate.add_trace(
                            go.Scatter(x=s["月份"], y=s["达标率"], name="达标率",
                                       mode="lines+markers", line=dict(color="#1565c0"),
                                       legendgroup=ind, showlegend=(i == 1)),
                            row=i, col=1, secondary_y=True)
                    fig_rate.update_layout(barmode="stack", height=320 * len(_rate_inds),
                                           xaxis_tickformat="%Y-%m",
                                           legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
                    for i in range(1, len(_rate_inds) + 1):
                        fig_rate.update_yaxes(title_text="人数", row=i, col=1)
                    fig_rate.update_yaxes(title_text="达标率", secondary_y=True, row=1, col=1,
                                          tickformat=".0%")
                    st.plotly_chart(fig_rate, width="stretch")
                    _download_csv(
                        rate_df2.assign(月份=rate_df2["月份"].dt.strftime("%Y-%m"),
                                        达标率=rate_df2["达标率"].map(lambda x: f"{x:.1%}")),
                        "单药房_规范治疗率.csv", "⬇ 下载规范治疗率", "dl_sp_rate")
                else:
                    st.info("该药房在所选时间范围内无规范治疗量可计算（适应症未匹配标准量）。")
            else:
                st.info("该药房在所选时间范围内无规范治疗率数据。")

            # 本药房一键打包下载（Excel）
            _rate_out = _fmt_month(rate_df2)
            if _rate_out is not None and "达标率" in _rate_out.columns:
                _rate_out = _rate_out.copy()
                _rate_out["达标率"] = _rate_out["达标率"].map(
                    lambda x: f"{x:.1%}" if pd.notna(x) else "")
            _sp_sheets = [
                (_fmt_month(ind_df), "适应症分布"),
                (_fmt_month(sales_df), "新老患销量"),
                (_fmt_month(np_df2), "新患趋势"),
                (_fmt_month(dot_df2), "R12M_DOT趋势"),
                (_rate_out, "规范治疗率"),
            ]
            _sp_data = _write_excel_bytes(_sp_sheets)
            st.download_button(
                label="⬇ 下载本药房全部数据（Excel）",
                data=_sp_data,
                file_name=f"{sel_single}_全部数据.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_sp_all",
            )

        # ---- 单药房自动总结 ----
        if len(sales_df):
            _tot_box = int(sales_df["老患销量"].sum() + sales_df["新患销量"].sum())
            _np_total = int(np_df2["新患数"].sum()) if len(np_df2) else 0
            _last_dot = dot_df2.iloc[-1]["DOT"] if len(dot_df2) else 0
            _summary_box(
                f"{sel_single} 在所选区间净销量合计 {_tot_box} 盒，新患 {_np_total} 人；"
                f"末月 R12M DOT 约 {_last_dot:.2f} 盒/人。"
            )

    with tab_clinical:
        # ============ 临床分析 - TOP5 医院 ============
        _tp = pd.Timestamp(target_month).to_period("M")
        _ml = [
            (_tp - 2).to_timestamp().strftime("%Y-%m"),
            (_tp - 1).to_timestamp().strftime("%Y-%m"),
            _tp.to_timestamp().strftime("%Y-%m"),
        ]
        st.subheader("🏨 临床分析 - TOP5 医院")
        st.caption(
            f"锚定月：{target_month.strftime('%Y年%m月')}；最近3个月 = {_ml[0]} ~ {_ml[2]}"
            f"（含锚定月及前两个月，跟随侧边栏「分布图目标年月」）。"
            "TOP5 按该3个月累计净销量排名；新患为全局首次购药患者；"
            "R12M DOT 按锚定月回滚12个月；25年DOT 为2025全年固定基线；同比 = (R12M − 25年) / 25年。"
        )

        # ---- 医院汇总表 ----
        st.subheader("TOP5 医院汇总")
        if len(hosp_df):
            _hosp_cfg = {
                "R12M DOT": st.column_config.NumberColumn(format="%.2f"),
                "25年DOT": st.column_config.NumberColumn(format="%.2f"),
                "同比": st.column_config.NumberColumn(format="%.1%"),
            }
            st.dataframe(hosp_df, column_config=_hosp_cfg, use_container_width=True, hide_index=True)
            _download_csv(hosp_df, "临床_TOP5医院.csv", "⬇ 下载 TOP5 医院汇总数据", "dl_clin_hosp")
        else:
            st.info("当前锚定月无医院销售数据。")

        # ---- 医生明细 ----
        st.divider()
        st.subheader("医生明细（每家 TOP3，近两个月）")
        st.caption(
            f"每家医院展示 TOP3 医生（按 {_ml[2]} 月销量排序）；列 = {_ml[1]} / {_ml[2]} 销量、患者数及环比（增长率）。"
            "医生姓名已脱敏。"
        )
        if len(doctor_df):
            _doc_cfg = {
                "销量环比": st.column_config.NumberColumn(format="%.1%"),
                "患者数环比": st.column_config.NumberColumn(format="%.1%"),
            }
            st.dataframe(doctor_df, column_config=_doc_cfg, use_container_width=True, hide_index=True)
            _download_csv(doctor_df, "临床_医生明细.csv", "⬇ 下载医生明细数据", "dl_clin_doc")
        else:
            st.info("当前范围内无医生明细数据。")

        # ---- 临床自动总结 ----
        if len(hosp_df):
            _sales_cols = [c for c in hosp_df.columns if "销量" in c]
            _cum = hosp_df[_sales_cols].sum(axis=1)
            _top = hosp_df.loc[_cum.idxmax()]
            _best = hosp_df.loc[hosp_df["同比"].idxmax()]
            _summary_box(
                f"TOP5 医院中，近 3 个月累计净销量最高的是「{_top['医院']}」（{_top[_sales_cols].sum():.0f} 盒）；"
                f"同比最好的是「{_best['医院']}」（{_best['同比']:.1%}）。"
            )


    with tab_ops:
        # ============ 运营分析 ============
        st.subheader("🛠 运营分析")
        st.caption("运营视角：各适应症全局对比、患者留存 cohort、断药预警。锚定月跟随侧边栏「分布图目标年月」。")

        # ---- 子模块1：适应症全局对比 ----
        st.subheader("① 各适应症全局对比（截至展示结束月）")
        st.caption("患者按全量末次购药适应症(清洗)整体归因；累计患者数=全量去重；窗口新患=首购月落在展示区间；"
                   "R12M DOT / 规范治疗率=展示结束月口径。")
        if len(ind_sum):
            _ind_cfg = {
                "规范治疗率": st.column_config.NumberColumn(format="%.1%"),
                "R12M DOT": st.column_config.NumberColumn(format="%.2f"),
            }
            st.dataframe(ind_sum, column_config=_ind_cfg, use_container_width=True, hide_index=True)
            _download_csv(ind_sum, "运营_适应症对比.csv", "⬇ 下载适应症全局对比", "dl_ops_ind")
            _top_ind = ind_sum.iloc[0]
            _best_rate = ind_sum.loc[ind_sum["规范治疗率"].idxmax()]
            _best_dot = ind_sum.loc[ind_sum["R12M DOT"].idxmax()]
            _summary_box(
                f"共覆盖 {len(ind_sum)} 个标准适应症，累计患者数最多的是「{_top_ind['标准适应症']}」"
                f"（{int(_top_ind['累计患者数'])} 人）。规范治疗率最高为「{_best_rate['标准适应症']}」"
                f"（{_best_rate['规范治疗率']:.1%}），R12M DOT 最高为「{_best_dot['标准适应症']}」"
                f"（{_best_dot['R12M DOT']:.2f} 盒/人）。"
            )
        else:
            st.info("无适应症数据。")

        # ---- 子模块2：留存 cohort ----
        st.divider()
        st.subheader("② 患者留存 cohort（按首购月）")
        st.caption("每行一个首购月 cohort；+k月留存=该批患者在首购后第 k 个月仍有购买（净销量>0）的占比。"
                   "颜色越深留存越好；灰=尚无数据（未到该月）。")
        if len(cohort):
            _cohort_cols = [c for c in cohort.columns if "留存" in c]
            _cohort_cfg = {c: st.column_config.NumberColumn(format="%.1%") for c in _cohort_cols}
            _heat = cohort.set_index("首购月")[_cohort_cols].T
            fig_coh = px.imshow(_heat, aspect="auto", color_continuous_scale="Blues",
                                labels=dict(x="首购月", y="距首购月数", color="留存率"),
                                title="留存率热力图")
            fig_coh.update_layout(height=360)
            st.plotly_chart(fig_coh, width="stretch")
            st.dataframe(cohort, column_config=_cohort_cfg, use_container_width=True, hide_index=True)
            _download_csv(cohort, "运营_留存漏斗.csv", "⬇ 下载留存 cohort", "dl_ops_cohort")
            _has12 = cohort[cohort["+12月留存"].notna()]
            if len(_has12):
                _c = _has12.iloc[0]
                _summary_box(
                    f"最早可追溯 12 个月留存的 cohort 为首购于 {_c['首购月']} 的 {int(_c['首购患者数'])} 名患者，"
                    f"其 +1 / +3 / +6 / +12 月留存分别为 "
                    f"{_c['+1月留存']:.1%} / {_c['+3月留存']:.1%} / {_c['+6月留存']:.1%} / {_c['+12月留存']:.1%}。"
                )
            else:
                _summary_box("当前所选区间暂无可追溯 +12 月留存的 cohort（需首购月不晚于数据末月前 12 个月）。")
        else:
            st.info("当前区间无首购患者。")

        # ---- 子模块3：断药预警 ----
        st.divider()
        st.subheader("③ 断药预警名单（末次购药距今 ≥ 3 个月）")
        st.caption("历史有购买、且末次购买月早于「锚定月 − 3 个月」的活跃患者。患者标识已脱敏，仅供运营筛选高危人群。")
        if len(churn):
            st.dataframe(churn, use_container_width=True, hide_index=True)
            _download_csv(churn, "运营_断药预警.csv", "⬇ 下载断药预警名单", "dl_ops_churn")
            _long = int((churn["断药月数"] >= 6).sum())
            _summary_box(
                f"共 {len(churn)} 名患者已断药（≥3 个月未购），其中 {_long} 名断药 ≥6 个月，需优先干预；"
                f"断药最久者已达 {int(churn['断药月数'].max())} 个月。"
            )
        else:
            st.info("当前锚定月下无断药预警患者（所有活跃患者末次购药均在 3 个月内）。")


if __name__ == "__main__":
    main()
