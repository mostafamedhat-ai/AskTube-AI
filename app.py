import os

import requests
import streamlit as st


st.set_page_config(
    page_title="Ask Tube",
    page_icon=":material/video_library:",
    layout="centered",
)


def get_setting(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except (FileNotFoundError, KeyError):
        value = None
    return value or os.getenv(name)

API_BASE_URL = get_setting("API_BASE_URL")
API_KEY = get_setting("API_KEY")

API_BASE_URL = API_BASE_URL.strip().rstrip("/")
API_KEY = API_KEY.strip()

if not API_BASE_URL or not API_KEY:
    st.error(
        "Set API_BASE_URL and API_KEY in Streamlit secrets or environment variables."
    )
    st.stop()

API_BASE_URL = API_BASE_URL.rstrip("/")
HEADERS = {"Authorization": f"Bearer {API_KEY}"}


def post_to_api(path: str, payload: dict, timeout: int) -> dict:
    try:
        response = requests.post(
            f"{API_BASE_URL}/{path.lstrip('/')}",
            headers=HEADERS,
            json=payload,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Could not connect to the API: {exc}") from exc

    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"API error ({response.status_code}): {detail}")

    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError("The API returned an invalid JSON response.") from exc


st.session_state.setdefault("video_url", "")
st.session_state.setdefault("video_id", None)
st.session_state.setdefault("summary", "")
st.session_state.setdefault("messages", [])

st.title("Ask Tube", icon=":material/video_library:")
st.caption("Load a YouTube video, review its summary, and ask questions about it.")

with st.container(border=True):
    st.subheader("Load a video")
    with st.form("load_video_form"):
        video_url_input = st.text_input(
            "YouTube video URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="The video needs an available English or Arabic transcript.",
        )
        load_submitted = st.form_submit_button(
            "Process video",
            type="primary",
            icon=":material/arrow_forward:",
            width="stretch",
        )

    if load_submitted:
        if not video_url_input.strip():
            st.warning("Paste a YouTube video URL first.")
        else:
            try:
                with st.spinner("Fetching transcript, building search index, and writing summary…"):
                    result = post_to_api(
                        "/LOAD_VIDEO",
                        {"video_url": video_url_input.strip()},
                        timeout=900,
                    )
                st.session_state.video_url = video_url_input.strip()
                st.session_state.video_id = result["video_id"]
                st.session_state.summary = result["summary"]
                st.session_state.messages = []
                st.success("Video is ready.")
            except (RuntimeError, KeyError) as exc:
                st.error(str(exc))

if st.session_state.video_id:
    st.subheader("Your video")
    st.video(st.session_state.video_url)

    with st.container(border=True):
        st.subheader("Video summary")
        st.write(st.session_state.summary)

    st.subheader("Ask about the video")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about this video…", submit_mode="disable")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching the transcript…"):
                    result = post_to_api(
                        "/ASK_TUBE",
                        {"question": question, "top_k": 4},
                        timeout=180,
                    )
                answer = result.get("response", "The API returned no answer.")
                if isinstance(answer, list):
                    answer = "\n".join(
                        item.get("text", "")
                        for item in answer
                        if isinstance(item, dict)
                    )
                answer = str(answer)
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            except (RuntimeError, KeyError) as exc:
                st.error(str(exc))
else:
    st.info("Load a video to see its summary and start asking questions.")

st.caption("This prototype keeps one active video in the backend at a time.")
