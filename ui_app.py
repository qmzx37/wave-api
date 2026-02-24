import streamlit as st
import requests
import json
from requests.exceptions import ReadTimeout, ConnectTimeout, ConnectionError

API_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(page_title="다모여 추천 UI", layout="centered")
st.title("📍 다모여 장소 추천 (Streamlit UI)")

# -------------------------
# 입력/옵션
# -------------------------
user_text = st.text_input("요청 문장", "광안리 조용한 카페 추천해줘")

col1, col2, col3 = st.columns(3)
avoid_franchise = col1.checkbox("프랜차이즈 제외", value=True)
stable_top5_sec = col2.slider("TOP5 고정(초)", 0, 60, 20)
balance_types = col3.checkbox("타입 균형(뷰/빵/디저트)", value=True)

window = st.slider("wave window", 3, 50, 10)

# ✅ 네트워크 옵션
with st.expander("⚙️ 네트워크 옵션", expanded=False):
    timeout_sec = st.slider("API timeout (초)", 10, 120, 60, 5)
    retry_once = st.checkbox("타임아웃이면 1회 재시도", value=True)

# -------------------------
# 유틸: 후보 키 호환
# -------------------------
def pick_candidate_fields(p: dict) -> dict:
    """
    FastAPI에서 내려오는 후보 dict 키가 여러 버전일 수 있어서
    name/addr/url/phone/category를 최대한 호환해서 뽑는다.
    """
    def g(*keys, default=""):
        for k in keys:
            v = p.get(k)
            if v is not None and str(v).strip() != "":
                return v
        return default

    return {
        "name": g("name", "place_name", "title"),
        "addr": g("addr", "address", "address_name", "road_address_name"),
        "phone": g("phone", "tel"),
        "url": g("url", "place_url", "link"),
        "category": g("category", "category_name"),
        "id": g("id", "place_id"),
    }

# -------------------------
# 호출
# -------------------------
if st.button("추천받기"):
    payload = {
        "text": user_text,
        "window": window,
        "avoid_franchise": avoid_franchise,
        "stable_top5_sec": stable_top5_sec,
        "balance_types": balance_types,
    }

    data = None
    status = None
    last_err = None

    with st.spinner("서버에 요청 중... (조금 오래 걸릴 수 있어)"):
        # ✅ 최대 2번: (1) 본요청 + (2) 타임아웃이면 1회 재시도
        for attempt in range(2):
            try:
                r = requests.post(API_URL, json=payload, timeout=timeout_sec)
                status = r.status_code

                # JSON 파싱 안전 처리
                try:
                    data = r.json()
                except Exception:
                    st.error("API가 JSON이 아닌 응답을 줬어.")
                    st.code(r.text)
                    st.stop()

                break  # 성공
            except (ReadTimeout, ConnectTimeout) as e:
                last_err = e
                if retry_once and attempt == 0:
                    continue
                else:
                    st.error(f"API 타임아웃: {repr(e)} (timeout={timeout_sec}s)")
                    st.stop()
            except ConnectionError as e:
                st.error(f"API 연결 실패(서버 꺼짐/포트 문제 가능): {repr(e)}")
                st.stop()
            except Exception as e:
                st.error(f"API 연결 실패: {repr(e)}")
                st.stop()

    if status != 200:
        st.error(f"API 응답 오류: status={status}")
        st.code(json.dumps(data, ensure_ascii=False, indent=2) if isinstance(data, dict) else str(data))
        st.stop()

    # -------------------------
    # 기본 출력
    # -------------------------
    st.success("추천 완료")

    st.subheader("🤖 reply")
    st.write(data.get("reply", ""))

    # -------------------------
    # TOP5 카드 표시
    #   - 너 서버 디버그 구조가 debug.kakao_place.candidates 또는
    #     debug.kakao_place.candidates / debug.kakao_place... 등 버전이 흔들려도 대응
    # -------------------------
    debug = (data.get("debug", {}) or {})
    kakao_place = (debug.get("kakao_place", {}) or {})

    # ✅ 서버가 candidates를 어디에 두든 최대한 찾아봄
    candidates = kakao_place.get("candidates")
    if candidates is None:
        # 혹시 debug["kakao_place"]["kakao_debug"] 같은 하위에 넣었을 수도 있어서 추가 탐색
        candidates = (debug.get("kakao_place", {}) or {}).get("candidates", [])
    candidates = candidates or []

    st.subheader("📍 TOP5 (카카오)")
    if not candidates:
        st.warning("TOP5가 비어있어. (현재 candidates가 없음) → debug에서 kakao_debug/meta/쿼리 확인해봐.")
    else:
        for i, raw in enumerate(candidates, 1):
            p = pick_candidate_fields(raw if isinstance(raw, dict) else {})
            name = p["name"]
            addr = p["addr"]
            phone = p["phone"]
            url = p["url"]
            category = p["category"]

            with st.container():
                st.markdown(f"### {i}. {name if name else '(이름 없음)'}")
                if category:
                    st.caption(category)
                if addr:
                    st.write("📍", addr)
                if phone:
                    st.write("📞", phone)
                if url:
                    st.link_button("카카오맵 열기", url)
                st.divider()

    # -------------------------
    # 축/웨이브 요약
    # -------------------------
    st.subheader("🧭 axes")
    st.json(data.get("axes", {}))

    st.subheader("🌊 wave")
    st.json(data.get("wave", {}))

    # -------------------------
    # 디버그 펼치기
    # -------------------------
    with st.expander("🧪 debug 전체 보기"):
        st.json(debug)
