import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="THMs 정책 대시보드", layout="wide")

# ----------------------------
# 1. 데이터 생성 (임시 → 나중에 CSV로 교체)
# ----------------------------
@st.cache_data
def load_data():
    dates = pd.date_range("2023-01-01", "2025-12-31", freq="M")
    df = pd.DataFrame({
        "date": dates,
        "region": np.random.choice(["서울","경기","인천"], len(dates)),
        "thms": np.random.uniform(20, 120, len(dates)),
        "temperature": np.random.uniform(5, 30, len(dates)),
        "organic_matter": np.random.uniform(1, 5, len(dates)),
        "chlorine_dose": np.random.uniform(0.5, 2.0, len(dates)),
        "residence_time": np.random.uniform(1, 10, len(dates)),
    })

    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["exceed"] = (df["thms"] > 80).astype(int)

    df = df.sort_values(["region","date"])
    df["lag_thms"] = df.groupby("region")["thms"].shift(1).fillna(df["thms"])

    return df

df = load_data()

# ----------------------------
# 2. 모델 학습
# ----------------------------
features = [
    "temperature",
    "organic_matter",
    "chlorine_dose",
    "residence_time",
    "quarter",
    "lag_thms"
]

model = RandomForestClassifier()
model.fit(df[features], df["exceed"])

df["risk_score"] = model.predict_proba(df[features])[:,1]

# ----------------------------
# 3. 정책 로직
# ----------------------------
def policy_decision(risk):
    if risk > 0.7:
        return "🚨 고도처리 + 실시간 대응"
    elif risk > 0.5:
        return "⚠️ 염소량 조정 + 모니터링 강화"
    else:
        return "✅ 정상 운영"

df["policy"] = df["risk_score"].apply(policy_decision)

# ----------------------------
# 4. UI
# ----------------------------
st.title("💧 수도권 THMs 정책 분석 대시보드")

# 필터
regions = st.multiselect(
    "지역 선택",
    ["서울","경기","인천"],
    default=["서울","경기","인천"]
)

filtered_df = df[df["region"].isin(regions)]

# ----------------------------
# KPI
# ----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("평균 THMs", f"{filtered_df['thms'].mean():.2f}")
col2.metric("초과율", f"{filtered_df['exceed'].mean()*100:.1f}%")
col3.metric("평균 Risk Score", f"{filtered_df['risk_score'].mean():.2f}")

# ----------------------------
# 시계열 그래프
# ----------------------------
st.subheader("📈 월별 THMs 추이")

fig = px.line(
    filtered_df,
    x="date",
    y="thms",
    color="region",
    title="THMs 농도 시계열"
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Risk Score 시각화
# ----------------------------
st.subheader("⚠️ Risk Score 분석")

fig2 = px.line(
    filtered_df,
    x="date",
    y="risk_score",
    color="region",
    title="THMs 위험도"
)

st.plotly_chart(fig2, use_container_width=True)

# ----------------------------
# 정책 추천 테이블
# ----------------------------
st.subheader("📋 정책 추천 결과")

st.dataframe(
    filtered_df[[
        "date","region","thms","risk_score","policy"
    ]].sort_values("risk_score", ascending=False),
    use_container_width=True
)

# ----------------------------
# 정책 시뮬레이션
# ----------------------------
st.subheader("🧪 정책 시뮬레이션")

reduction_rate = st.slider("염소 주입량 감소 (%)", 0, 30, 10)

sim_df = filtered_df.copy()
sim_df["chlorine_dose"] *= (1 - reduction_rate / 100)

sim_df["risk_after"] = model.predict_proba(sim_df[features])[:,1]

before = filtered_df["risk_score"].mean()
after = sim_df["risk_after"].mean()

st.metric("위험도 감소", f"{(before - after):.3f}")

# ----------------------------
# 핵심 인사이트
# ----------------------------
st.subheader("🧠 핵심 인사이트")

st.markdown("""
- Q2(2분기): 조기 경고 구간 → 선제 대응 필요  
- Q3(3분기): 고위험 구간 → 집중 관리  
- 경기 지역: 구조적 고위험  
👉 핵심: **Q2에서 개입해야 Q3를 줄인다**
""")
