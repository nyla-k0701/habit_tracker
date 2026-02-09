# app.py
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

# OpenAI SDK (Responses API)
# pip install openai
from openai import OpenAI

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="AI 습관 트래커", page_icon="📊", layout="wide")
st.title("📊 AI 습관 트래커")


# ----------------------------
# Helpers / APIs
# ----------------------------
def _safe_get_json(url: str, timeout: int = 10, params: Optional[dict] = None) -> Optional[dict]:
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def get_weather(city: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    OpenWeatherMap에서 날씨 가져오기 (한국어, 섭씨).
    실패 시 None 반환. timeout=10
    """
    if not api_key:
        return None

    data = _safe_get_json(
        "https://api.openweathermap.org/data/2.5/weather",
        timeout=10,
        params={"q": city, "appid": api_key, "units": "metric", "lang": "kr"},
    )
    if not data:
        return None

    try:
        weather_desc = (data.get("weather") or [{}])[0].get("description")
        temp = (data.get("main") or {}).get("temp")
        feels = (data.get("main") or {}).get("feels_like")
        humidity = (data.get("main") or {}).get("humidity")
        icon = (data.get("weather") or [{}])[0].get("icon")
        return {
            "city": city,
            "description": weather_desc,
            "temp_c": temp,
            "feels_like_c": feels,
            "humidity": humidity,
            "icon": icon,
        }
    except Exception:
        return None


def _extract_breed_from_dog_url(url: str) -> Optional[str]:
    # Dog CEO URL 예: https://images.dog.ceo/breeds/hound-afghan/n02088094_1003.jpg
    m = re.search(r"/breeds/([^/]+)/", url)
    if not m:
        return None
    raw = m.group(1)  # e.g. "hound-afghan" or "retriever-golden"
    # "hound-afghan" -> "hound afghan" (조금 더 보기 좋게)
    return raw.replace("-", " ").strip()


def get_dog_image() -> Optional[Dict[str, Any]]:
    """
    Dog CEO에서 랜덤 강아지 사진 URL과 품종 가져오기.
    실패 시 None 반환. timeout=10
    """
    data = _safe_get_json("https://dog.ceo/api/breeds/image/random", timeout=10)
    if not data or data.get("status") != "success":
        return None

    try:
        url = data.get("message")
        breed = _extract_breed_from_dog_url(url) or "unknown"
        return {"image_url": url, "breed": breed}
    except Exception:
        return None


def _coach_system_prompt(style: str) -> str:
    base = (
        "너는 사용자의 일상 습관 체크인을 바탕으로 짧고 실용적인 '컨디션 리포트'를 작성하는 코치다.\n"
        "절대 과장하지 말고, 한국어로, 친근하지만 명확하게 말한다.\n"
        "의학적 진단/치료 조언은 하지 말고, 생활 습관 관점의 일반적 조언만 제공한다.\n"
        "아래 출력 형식을 정확히 지켜라.\n"
        "\n"
        "출력 형식(반드시 이 순서/헤더 유지):\n"
        "1) 컨디션 등급: [S/A/B/C/D]\n"
        "2) 습관 분석: (2~5줄)\n"
        "3) 날씨 코멘트: (1~2줄)\n"
        "4) 내일 미션: (불릿 3개)\n"
        "5) 오늘의 한마디: (짧게 1줄)\n"
    )

    if style == "스파르타 코치":
        return base + "\n추가 스타일: 엄격하고 직설적. 핑계 차단. 행동 지시 위주."
    if style == "따뜻한 멘토":
        return base + "\n추가 스타일: 따뜻하고 공감. 작은 성취를 인정하며 현실적인 다음 कदम 제안."
    if style == "게임 마스터":
        return base + "\n추가 스타일: RPG/퀘스트 느낌. 레벨/보상/던전 같은 표현을 적절히 섞되 유치하지 않게."
    return base


def generate_report(
    openai_api_key: str,
    coach_style: str,
    date_str: str,
    habits_checked: List[str],
    mood: int,
    weather: Optional[Dict[str, Any]],
    dog: Optional[Dict[str, Any]],
    achievement_rate: int,
) -> Optional[str]:
    """
    습관+기분+날씨+강아지 품종을 모아서 OpenAI에 전달.
    모델: gpt-5-mini
    실패 시 None 반환.
    """
    if not openai_api_key:
        return None

    weather_text = "날씨 정보 없음"
    if weather:
        weather_text = (
            f"{weather.get('city')} / {weather.get('description')} / "
            f"{weather.get('temp_c')}°C (체감 {weather.get('feels_like_c')}°C) / 습도 {weather.get('humidity')}%"
        )

    dog_text = "강아지 정보 없음"
    if dog:
        dog_text = f"품종: {dog.get('breed')} / 이미지: {dog.get('image_url')}"

    habits_text = ", ".join(habits_checked) if habits_checked else "없음"

    user_input = (
        f"날짜: {date_str}\n"
        f"달성률: {achievement_rate}%\n"
        f"완료한 습관: {habits_text}\n"
        f"기분(1~10): {mood}\n"
        f"날씨: {weather_text}\n"
        f"강아지: {dog_text}\n"
        "\n"
        "요청: 위 정보를 바탕으로 출력 형식에 맞춰 컨디션 리포트를 작성해줘."
    )

    try:
        client = OpenAI(api_key=openai_api_key)
        resp = client.responses.create(
            model="gpt-5-mini",
            instructions=_coach_system_prompt(coach_style),
            input=user_input,
        )
        text = getattr(resp, "output_text", None)
        return text.strip() if text else None
    except Exception:
        return None


# ----------------------------
# Sidebar: API keys
# ----------------------------
with st.sidebar:
    st.header("🔑 API 설정")
    openai_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    owm_key = st.text_input("OpenWeatherMap API Key", type="password", placeholder="OWM Key...")
    st.caption("키는 로컬에서만 사용되며 앱이 저장하지 않도록 주의하세요.")


# ----------------------------
# Session state: initialize
# ----------------------------
@dataclass
class DayRecord:
    date: str  # YYYY-MM-DD
    rate: int
    checked_count: int
    mood: int


def _init_demo_data() -> List[DayRecord]:
    today = dt.date.today()
    # 데모용 6일 샘플 데이터 (오늘 제외)
    samples = []
    # 적당히 그럴듯한 패턴
    demo = [
        (6, 60, 3, 6),
        (5, 80, 4, 7),
        (4, 40, 2, 5),
        (3, 100, 5, 8),
        (2, 60, 3, 6),
        (1, 80, 4, 7),
    ]
    for days_ago, rate, checked, mood in demo:
        d = (today - dt.timedelta(days=days_ago)).isoformat()
        samples.append(DayRecord(date=d, rate=rate, checked_count=checked, mood=mood))
    return samples


if "records" not in st.session_state:
    st.session_state.records = _init_demo_data()  # List[DayRecord]


# ----------------------------
# Check-in UI
# ----------------------------
st.subheader("✅ 오늘의 체크인")

CITIES = [
    "Seoul",
    "Busan",
    "Incheon",
    "Daegu",
    "Daejeon",
    "Gwangju",
    "Ulsan",
    "Suwon",
    "Jeju",
    "Sejong",
]

COACH_STYLES = ["스파르타 코치", "따뜻한 멘토", "게임 마스터"]

HABITS = [
    ("🌅", "기상 미션"),
    ("💧", "물 마시기"),
    ("📚", "공부/독서"),
    ("🏃", "운동하기"),
    ("😴", "수면"),
]

col_left, col_right = st.columns([1.2, 1])
with col_left:
    c1, c2 = st.columns(2)
    checks = {}
    # 2열 배치 (3 + 2)
    with c1:
        for emoji, label in HABITS[:3]:
            checks[label] = st.checkbox(f"{emoji} {label}", key=f"habit_{label}")
    with c2:
        for emoji, label in HABITS[3:]:
            checks[label] = st.checkbox(f"{emoji} {label}", key=f"habit_{label}")

    mood = st.slider("🙂 오늘 기분은 어때요?", 1, 10, 6, help="1=최악, 10=최고")
with col_right:
    city = st.selectbox("🏙️ 도시 선택", CITIES, index=0)
    coach_style = st.radio("🎭 코치 스타일", COACH_STYLES, horizontal=False)

checked_habits = [h for h, v in checks.items() if v]
checked_count = len(checked_habits)
achievement_rate = int(round((checked_count / len(HABITS)) * 100))


# ----------------------------
# Metrics
# ----------------------------
m1, m2, m3 = st.columns(3)
m1.metric("달성률", f"{achievement_rate}%")
m2.metric("달성 습관", f"{checked_count} / {len(HABITS)}")
m3.metric("기분", f"{mood} / 10")


# ----------------------------
# Records + chart (7 days)
# ----------------------------
def _upsert_today_record():
    today_str = dt.date.today().isoformat()
    # 동일 날짜 있으면 갱신, 없으면 추가
    records: List[DayRecord] = st.session_state.records
    for i, r in enumerate(records):
        if r.date == today_str:
            records[i] = DayRecord(
                date=today_str,
                rate=achievement_rate,
                checked_count=checked_count,
                mood=mood,
            )
            return
    records.append(
        DayRecord(
            date=today_str,
            rate=achievement_rate,
            checked_count=checked_count,
            mood=mood,
        )
    )


# 오늘 값은 UI 변경마다 차트에 반영되도록 upsert
_upsert_today_record()

# 최근 7일만 정렬 후 사용
records_sorted = sorted(st.session_state.records, key=lambda x: x.date)
records_last7 = records_sorted[-7:]

df = pd.DataFrame(
    [{"date": r.date, "achievement_rate": r.rate, "mood": r.mood, "checked": r.checked_count} for r in records_last7]
)
df["date"] = pd.to_datetime(df["date"])

st.subheader("📈 최근 7일 달성률")
st.bar_chart(df.set_index("date")[["achievement_rate"]], height=260)


# ----------------------------
# Generate report section
# ----------------------------
st.divider()
st.subheader("🧠 AI 코치 리포트")

today_str = dt.date.today().isoformat()

btn = st.button("컨디션 리포트 생성", type="primary", use_container_width=True)

weather: Optional[Dict[str, Any]] = None
dog: Optional[Dict[str, Any]] = None
report_text: Optional[str] = None

if btn:
    with st.spinner("날씨/강아지/AI 리포트를 준비 중..."):
        weather = get_weather(city, owm_key)
        dog = get_dog_image()

        report_text = generate_report(
            openai_api_key=openai_key,
            coach_style=coach_style,
            date_str=today_str,
            habits_checked=checked_habits,
            mood=mood,
            weather=weather,
            dog=dog,
            achievement_rate=achievement_rate,
        )

    # 2열 카드: 날씨 + 강아지
    wcol, dcol = st.columns(2)

    with wcol:
        st.markdown("#### ☁️ 오늘의 날씨")
        if weather:
            icon = weather.get("icon")
            icon_url = f"https://openweathermap.org/img/wn/{icon}@2x.png" if icon else None
            if icon_url:
                st.image(icon_url, width=80)
            st.write(f"**도시:** {weather.get('city')}")
            st.write(f"**상태:** {weather.get('description')}")
            st.write(f"**기온:** {weather.get('temp_c')}°C (체감 {weather.get('feels_like_c')}°C)")
            st.write(f"**습도:** {weather.get('humidity')}%")
        else:
            st.info("날씨 정보를 불러오지 못했어요. (API Key/도시/네트워크 확인)")

    with dcol:
        st.markdown("#### 🐶 오늘의 강아지")
        if dog:
            st.image(dog.get("image_url"), use_container_width=True)
            st.write(f"**품종:** {dog.get('breed')}")
        else:
            st.info("강아지 이미지를 불러오지 못했어요. (네트워크 확인)")

    st.markdown("#### 📝 AI 리포트")
    if report_text:
        st.markdown(report_text)
    else:
        st.warning("리포트를 생성하지 못했어요. (OpenAI API Key/모델/네트워크 확인)")

    # 공유용 텍스트
    share_text = (
        f"📊 AI 습관 트래커 ({today_str})\n"
        f"- 달성률: {achievement_rate}% ({checked_count}/{len(HABITS)})\n"
        f"- 완료: {', '.join(checked_habits) if checked_habits else '없음'}\n"
        f"- 기분: {mood}/10\n"
        f"- 도시: {city}\n"
        f"- 코치: {coach_style}\n"
        f"- 날씨: {weather.get('description')} / {weather.get('temp_c')}°C" if weather else ""
    )
    st.markdown("#### 🔗 공유용 텍스트")
    st.code(share_text, language="text")


# ----------------------------
# Footer: API 안내
# ----------------------------
with st.expander("📌 API 안내 (필수 키/링크/주의사항)"):
    st.markdown(
        """
- **OpenAI API Key**: OpenAI 플랫폼에서 발급한 키가 필요합니다.  
  - 모델은 **`gpt-5-mini`** 를 사용합니다.
- **OpenWeatherMap API Key**: OpenWeatherMap에서 발급한 키가 필요합니다.  
  - 본 앱은 `units=metric`(섭씨), `lang=kr`(한국어 설명)으로 요청합니다.
- **Dog CEO API**: 키 없이 무료로 사용됩니다.
- **보안 팁**: 키를 코드에 하드코딩하지 말고, Streamlit Cloud 사용 시 `secrets` 또는 환경변수 사용을 권장합니다.
        """.strip()
    )
