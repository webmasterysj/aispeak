from __future__ import annotations

from uuid import uuid4

import streamlit as st
from groq import (
    APIError,
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    Groq,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)
from streamlit.errors import StreamlitSecretNotFoundError


MODEL_ID = "openai/gpt-oss-120b"
SYSTEM_PROMPT = (
    "당신은 한국어로 친절하고 명확하게 답하는 일상 대화형 AI 비서입니다. "
    "확인할 수 없는 실시간 정보는 사실처럼 단정하지 말고, 필요한 경우 한계를 밝히세요."
)
WELCOME_MESSAGE = "안녕하세요! 오늘도 무엇이든 편하게 물어보세요."
MISSING_KEY_MESSAGE = (
    "먼저 위의 Groq API 키 입력란에 API 키를 입력해 주세요."
)
SUGGESTIONS = (
    "오늘 하루를 알차게 보내는 방법을 알려줘.",
    "인공지능을 쉽게 설명해 줘.",
    "집중력을 높이는 방법이 궁금해.",
    "재미있는 상식 하나 알려줘.",
)


class RequestFailure(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def _read_api_key() -> str:
    try:
        value = st.secrets.get("GROQ_API_KEY", "")
    except StreamlitSecretNotFoundError:
        value = ""
    secret_key = value.strip() if isinstance(value, str) else ""
    if secret_key:
        return secret_key
    session_key = st.session_state.get("api_key_input", "")
    return session_key.strip() if isinstance(session_key, str) else ""


def _request_answer(api_key: str, messages: list[dict[str, str]]) -> str:
    client = Groq(api_key=api_key, timeout=30.0, max_retries=2)
    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
        )
    except AuthenticationError:
        raise RequestFailure(
            "Groq API 키가 올바른지 확인해 주세요."
        ) from None
    except PermissionDeniedError:
        raise RequestFailure(
            "AI 서비스 이용 권한을 확인할 수 없어요. 관리자에게 설정을 확인해 달라고 요청해 주세요."
        ) from None
    except (BadRequestError, NotFoundError):
        raise RequestFailure(
            "AI 서비스 설정을 확인해야 해요. 관리자에게 모델 설정을 확인해 달라고 요청해 주세요."
        ) from None
    except RateLimitError:
        raise RequestFailure(
            "요청이 많아 잠시 기다려야 해요. 잠시 후 다시 시도해 주세요."
        ) from None
    except (APITimeoutError, APIConnectionError, InternalServerError):
        raise RequestFailure(
            "지금은 AI 서비스에 연결하지 못했어요. 잠시 후 다시 시도해 주세요."
        ) from None
    except APIError:
        raise RequestFailure(
            "AI 서비스에서 요청을 처리하지 못했어요. 잠시 후 다시 시도해 주세요."
        ) from None

    if not response.choices:
        raise RequestFailure(
            "AI 서비스에서 답변을 받지 못했어요. 잠시 후 다시 시도해 주세요."
        )
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise RequestFailure(
            "AI 서비스에서 답변을 받지 못했어요. 잠시 후 다시 시도해 주세요."
        )
    return content.strip()


def _remember_event(event_id: str) -> bool:
    seen = st.session_state["seen_audio_event_ids"]
    if event_id in seen:
        return False
    seen.append(event_id)
    del seen[:-100]
    return True


def _submit_prompt(prompt: str, event_id: str | None = None) -> None:
    prompt = prompt.strip()
    if not prompt:
        return

    request_id = event_id or uuid4().hex
    if request_id == st.session_state["active_request_id"]:
        return
    if event_id is not None and not _remember_event(event_id):
        return

    st.session_state["active_request_id"] = request_id
    st.session_state["last_error"] = ""
    user_message = {
        "id": request_id,
        "role": "user",
        "content": prompt,
        "status": "pending",
    }
    st.session_state["messages"].append(user_message)

    api_key = _read_api_key()
    if not api_key:
        user_message["status"] = "failed"
        st.session_state["last_error"] = MISSING_KEY_MESSAGE
    else:
        api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        api_messages.extend(
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in st.session_state["messages"]
            if message["status"] == "complete"
        )
        api_messages.append({"role": "user", "content": prompt})
        try:
            answer = _request_answer(api_key, api_messages)
        except RequestFailure as exc:
            user_message["status"] = "failed"
            st.session_state["last_error"] = exc.message
        else:
            user_message["status"] = "complete"
            st.session_state["messages"].append(
                {
                    "id": uuid4().hex,
                    "role": "assistant",
                    "content": answer,
                    "status": "complete",
                }
            )
            st.session_state["speech_id"] = uuid4().hex
            st.session_state["speech_text"] = answer

    st.session_state["active_request_id"] = ""
    st.rerun()


def _reset_conversation() -> None:
    st.session_state["messages"] = []
    st.session_state["last_error"] = ""
    st.session_state["speech_id"] = ""
    st.session_state["speech_text"] = ""
    st.session_state["stop_speech_id"] = uuid4().hex
    st.session_state["active_request_id"] = ""


_VOICE_COMPONENT = st.components.v2.component(
    name="daily_speaker_voice_controls",
    html="""
<div class="voice-controls" role="group" aria-label="음성 컨트롤">
  <button id="listen-start" type="button">음성 입력 시작</button>
  <button id="listen-stop" type="button">음성 입력 종료</button>
  <button id="speak-again" type="button">답변 다시 듣기</button>
  <button id="speak-stop" type="button">답변 재생 중지</button>
  <p id="voice-status" role="status" aria-live="polite"></p>
</div>
""",
    css="""
.voice-controls {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem;
  color: var(--st-text-color);
  font-family: var(--st-font);
}
.voice-controls button {
  min-height: 2.75rem;
  padding: 0.5rem 0.8rem;
  border: 1px solid var(--st-border-color);
  border-radius: var(--st-border-radius);
  background: var(--st-secondary-background-color);
  color: var(--st-text-color);
  cursor: pointer;
}
.voice-controls button:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}
.voice-controls p {
  flex-basis: 100%;
  margin: 0;
  font-size: 0.875rem;
}
""",
    js="""
export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  const root = parentElement.querySelector(".voice-controls");
  if (!root) return;

  const startButton = root.querySelector("#listen-start");
  const stopButton = root.querySelector("#listen-stop");
  const replayButton = root.querySelector("#speak-again");
  const cancelButton = root.querySelector("#speak-stop");
  const status = root.querySelector("#voice-status");
  const state = root.__speakerState || (root.__speakerState = {
    recognition: null,
    lastSpeechId: "",
    stopId: data.stopSpeechId,
    voiceEnabled: data.voiceEnabled,
    utterance: null
  });
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  const synthesis = window.speechSynthesis;

  function setStatus(message) {
    status.textContent = message;
  }

  function cancelSpeech(message) {
    state.utterance = null;
    if (synthesis) synthesis.cancel();
    cancelButton.disabled = true;
    if (message) setStatus(message);
  }

  function speakAnswer() {
    if (!data.voiceEnabled) {
      setStatus("음성 답변이 꺼져 있어요.");
      return;
    }
    if (!synthesis || !data.speechText) {
      setStatus("이 환경에서는 음성 재생을 사용할 수 없어요. 답변은 텍스트로 확인해 주세요.");
      return;
    }
    cancelSpeech("");
    const utterance = new SpeechSynthesisUtterance(data.speechText);
    utterance.lang = "ko-KR";
    utterance.onstart = () => {
      if (state.utterance !== utterance) return;
      cancelButton.disabled = false;
      setStatus("답변을 읽고 있어요.");
    };
    utterance.onend = () => {
      if (state.utterance !== utterance) return;
      state.utterance = null;
      cancelButton.disabled = true;
      setStatus("답변 읽기가 끝났어요.");
    };
    utterance.onerror = (event) => {
      if (state.utterance !== utterance) return;
      state.utterance = null;
      cancelButton.disabled = true;
      setStatus(event.error === "canceled" || event.error === "interrupted"
        ? "답변 읽기를 중지했어요."
        : "자동 재생에 실패했어요. 다시 듣기 버튼을 눌러 보세요.");
    };
    state.utterance = utterance;
    cancelButton.disabled = false;
    setStatus("답변을 자동으로 읽으려고 해요. 재생되지 않으면 다시 듣기를 눌러 주세요.");
    synthesis.speak(utterance);
  }

  startButton.disabled = !Recognition;
  stopButton.disabled = !state.recognition;
  replayButton.disabled = !synthesis || !data.speechText || !data.voiceEnabled;
  cancelButton.disabled = !state.utterance;

  if (state.stopId !== data.stopSpeechId) {
    state.stopId = data.stopSpeechId;
    cancelSpeech("");
  }
  if (state.voiceEnabled && !data.voiceEnabled) {
    cancelSpeech("음성 답변이 꺼졌어요.");
  }
  state.voiceEnabled = data.voiceEnabled;

  if (!Recognition && !status.textContent) {
    setStatus("이 브라우저는 음성 입력을 지원하지 않아요. 텍스트 입력을 이용해 주세요.");
  } else if (!synthesis && !status.textContent) {
    setStatus("이 환경은 음성 읽기를 지원하지 않아요. 답변은 텍스트로 확인해 주세요.");
  } else if (!status.textContent) {
    setStatus("음성 입력은 버튼을 눌렀을 때 시작돼요. 마이크 권한이 필요합니다.");
  }

  startButton.onclick = () => {
    if (!Recognition) {
      setStatus("이 브라우저는 음성 입력을 지원하지 않아요. 텍스트 입력을 이용해 주세요.");
      return;
    }
    if (state.recognition) return;
    const recognition = new Recognition();
    state.recognition = recognition;
    recognition.lang = "ko-KR";
    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.onresult = (event) => {
      const transcript = Array.from(event.results)
        .map((result) => result[0].transcript)
        .join("")
        .trim();
      if (!transcript) {
        setStatus("음성을 알아듣지 못했어요. 다시 시도하거나 텍스트로 입력해 주세요.");
        return;
      }
      setStatus("음성을 인식했어요. 답변을 준비할게요.");
      setTriggerValue("recognized", {
        id: `${Date.now()}-${Math.random()}`,
        text: transcript
      });
    };
    recognition.onerror = (event) => {
      state.recognition = null;
      stopButton.disabled = true;
      setStatus(event.error === "not-allowed" || event.error === "service-not-allowed"
        ? "마이크 권한이 없어 음성 입력을 시작할 수 없어요. 권한을 허용하거나 텍스트로 입력해 주세요."
        : event.error === "audio-capture"
          ? "사용할 수 있는 마이크를 찾지 못했어요. 마이크를 확인하거나 텍스트로 입력해 주세요."
        : event.error === "no-speech"
          ? "음성이 들리지 않았어요. 다시 시도하거나 텍스트로 입력해 주세요."
          : event.error === "network"
            ? "음성 인식 서비스에 연결하지 못했어요. 다시 시도하거나 텍스트로 입력해 주세요."
          : "음성 입력 중 문제가 생겼어요. 다시 시도하거나 텍스트로 입력해 주세요.");
    };
    recognition.onend = () => {
      state.recognition = null;
      stopButton.disabled = true;
    };
    try {
      recognition.start();
      stopButton.disabled = false;
      setStatus("듣고 있어요. 말씀하신 뒤 입력이 끝나면 자동으로 전송됩니다.");
    } catch (error) {
      state.recognition = null;
      stopButton.disabled = true;
      setStatus(error.name === "NotAllowedError"
        ? "마이크 권한이 없어 음성 입력을 시작할 수 없어요. 권한을 허용하거나 텍스트로 입력해 주세요."
        : "음성 입력을 시작하지 못했어요. 텍스트 입력을 이용해 주세요.");
    }
  };

  stopButton.onclick = () => {
    if (state.recognition) {
      state.recognition.stop();
      state.recognition = null;
    }
    stopButton.disabled = true;
    setStatus("음성 입력을 종료했어요.");
  };
  replayButton.onclick = speakAnswer;
  cancelButton.onclick = () => cancelSpeech("답변 읽기를 중지했어요.");

  if (data.speechId && state.lastSpeechId !== data.speechId) {
    state.lastSpeechId = data.speechId;
    if (data.voiceEnabled) speakAnswer();
  }

  return () => {
    if (state.recognition) state.recognition.stop();
    if (synthesis) synthesis.cancel();
  };
}
""",
)


st.set_page_config(page_title="AI 스피커", page_icon=":material/graphic_eq:", layout="centered")
st.session_state.setdefault("messages", [])
st.session_state.setdefault("voice_enabled", True)
st.session_state.setdefault("speech_id", "")
st.session_state.setdefault("speech_text", "")
st.session_state.setdefault("stop_speech_id", "")
st.session_state.setdefault("seen_audio_event_ids", [])
st.session_state.setdefault("active_request_id", "")
st.session_state.setdefault("last_error", "")

st.title("AI 스피커")
st.caption("편하게 질문해 보세요. 대화는 현재 브라우저 세션에서만 유지됩니다.")

with st.expander("Groq API 키 입력", expanded=True):
    st.text_input(
        "Groq API 키",
        type="password",
        key="api_key_input",
        placeholder="gsk_로 시작하는 API 키 입력",
        help="입력한 키는 현재 앱 세션에서만 사용됩니다. 신뢰하는 앱에서만 입력하고, 코드에 직접 넣거나 공유하지 마세요.",
    )
    st.caption("키는 입력란에서 마스킹되며 대화에 표시되지 않습니다. 공용 또는 타인의 서버에서는 입력하지 마세요.")

with st.container(horizontal=True, vertical_alignment="center"):
    st.toggle("음성 답변 켜기", key="voice_enabled")
    st.button(
        "대화 초기화",
        key="reset_conversation",
        icon=":material/delete_sweep:",
        on_click=_reset_conversation,
    )

voice_result = _VOICE_COMPONENT(
    key="speaker_voice_controls",
    data={
        "voiceEnabled": st.session_state["voice_enabled"],
        "speechId": st.session_state["speech_id"],
        "speechText": st.session_state["speech_text"],
        "stopSpeechId": st.session_state["stop_speech_id"],
    },
    on_recognized_change=lambda: None,
    width="stretch",
)

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if not st.session_state["messages"]:
    with st.chat_message("assistant"):
        st.markdown(WELCOME_MESSAGE)

if st.session_state["last_error"]:
    st.error(st.session_state["last_error"])

st.caption("예시 질문")
example_prompt = None
first_row = st.columns(2)
second_row = st.columns(2)
for index, (column, suggestion) in enumerate(
    zip(first_row + second_row, SUGGESTIONS)
):
    with column:
        if st.button(
            suggestion,
            key=f"suggestion_{index}",
            width="stretch",
        ):
            example_prompt = suggestion

typed_prompt = st.chat_input(
    "메시지를 입력해 주세요",
    submit_mode="disable",
)

recognized = getattr(voice_result, "recognized", None)
if typed_prompt:
    _submit_prompt(typed_prompt)
elif example_prompt:
    _submit_prompt(example_prompt)
elif (
    isinstance(recognized, dict)
    and isinstance(recognized.get("id"), str)
    and isinstance(recognized.get("text"), str)
):
    _submit_prompt(recognized["text"], event_id=recognized["id"])
