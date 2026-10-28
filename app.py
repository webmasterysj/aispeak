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
    "모든 답변은 핵심만 담아 1~3문장 이내로 짧고 간결하게 답변하세요. "
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


_PAGE_STYLE = """
<style>
.stMainBlockContainer {
  max-width: 44rem;
  padding-top: 2rem;
  padding-bottom: 3rem;
}
h1 {
  text-align: center;
  font-weight: 800 !important;
  letter-spacing: -0.03em !important;
  margin-bottom: 0.15rem !important;
}
.stMainBlockContainer > div > div > div > .stCaptionContainer {
  text-align: center;
}
[data-testid="stExpander"] {
  border-radius: 1rem !important;
  border: 1px solid rgba(148, 163, 184, 0.25) !important;
  background: var(--st-secondary-background-color);
  box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
  overflow: hidden;
}
[data-testid="stChatMessage"] {
  border-radius: 1.15rem;
  padding: 0.85rem 1.1rem;
  margin-bottom: 0.65rem;
  border: 1px solid rgba(148, 163, 184, 0.18);
  box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
}
[data-testid="stBaseButton-secondary"] {
  border-radius: 0.9rem !important;
  border: 1px solid rgba(148, 163, 184, 0.28) !important;
  transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease !important;
}
[data-testid="stBaseButton-secondary"]:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08);
}
</style>
"""


_VOICE_COMPONENT = st.components.v2.component(
    name="daily_speaker_voice_controls",
    html="""
<div class="voice-controls" role="group" aria-label="음성 컨트롤" data-state="idle">
  <div class="speaker-stage">
    <span class="speaker-badge">SMART VOICE ASSISTANT</span>
    <div class="orb-wrapper">
      <button id="listen-start" class="orb-button" type="button" aria-label="음성 입력 시작">
        <svg class="mic-icon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
          <path d="M12 14a3 3 0 0 0 3-3V5a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.92V21h2v-3.08A7 7 0 0 0 19 11h-2z"/>
        </svg>
        <span class="orb-label">음성 입력 시작</span>
      </button>
    </div>
    <p id="voice-transcript" class="voice-transcript" aria-live="polite"></p>
    <p id="voice-status" role="status" aria-live="polite"></p>
  </div>
  <div class="sub-controls">
    <button id="listen-stop" type="button">음성 입력 종료</button>
    <button id="speak-again" type="button">답변 다시 듣기</button>
    <button id="speak-stop" type="button">답변 재생 중지</button>
  </div>
</div>
""",
    css="""
.voice-controls {
  box-sizing: border-box;
  width: 100%;
  padding: 1.75rem 1.35rem 1.35rem;
  border-radius: 1.65rem;
  background: radial-gradient(circle at 50% 22%, rgba(99, 102, 241, 0.22), transparent 60%),
              linear-gradient(145deg, #0b1120 0%, #1e293b 55%, #0f172a 100%);
  border: 1px solid rgba(255, 255, 255, 0.12);
  box-shadow: 0 20px 44px rgba(15, 23, 42, 0.26), inset 0 1px 0 rgba(255, 255, 255, 0.14);
  color: #f8fafc;
  font-family: var(--st-font, system-ui, -apple-system, sans-serif);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 1.1rem;
  text-align: center;
}
.speaker-stage {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.7rem;
  width: 100%;
}
.speaker-badge {
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.14em;
  color: #94a3b8;
  padding: 0.25rem 0.7rem;
  border-radius: 9999px;
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.1);
}
.orb-wrapper {
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0.65rem;
}
.orb-wrapper::before {
  content: "";
  position: absolute;
  width: 9.6rem;
  height: 9.6rem;
  border-radius: 50%;
  border: 1px solid rgba(129, 140, 248, 0.28);
  background: radial-gradient(circle, rgba(99, 102, 241, 0.18) 0%, transparent 70%);
  pointer-events: none;
  transition: transform 0.3s ease, border-color 0.3s ease;
}
.orb-button {
  position: relative;
  z-index: 2;
  width: 8.25rem;
  height: 8.25rem;
  border-radius: 50%;
  border: 2px solid rgba(255, 255, 255, 0.32);
  background: linear-gradient(135deg, #3b82f6 0%, #6366f1 52%, #8b5cf6 100%);
  color: #ffffff;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.4rem;
  cursor: pointer;
  box-shadow: 0 12px 32px rgba(99, 102, 241, 0.48), inset 0 2px 8px rgba(255, 255, 255, 0.38);
  transition: transform 0.22s ease, box-shadow 0.22s ease, background 0.3s ease;
  padding: 0.65rem;
}
.orb-button:hover:not(:disabled) {
  transform: translateY(-2px) scale(1.03);
  box-shadow: 0 16px 38px rgba(99, 102, 241, 0.62), inset 0 2px 10px rgba(255, 255, 255, 0.48);
}
.orb-button:active:not(:disabled) {
  transform: scale(0.96);
}
.orb-button:disabled {
  cursor: not-allowed;
  opacity: 0.52;
  box-shadow: none;
}
.mic-icon {
  width: 2.25rem;
  height: 2.25rem;
  fill: currentColor;
  flex-shrink: 0;
  filter: drop-shadow(0 2px 4px rgba(15, 23, 42, 0.25));
}
.orb-label {
  font-size: 0.82rem;
  font-weight: 700;
  letter-spacing: -0.01em;
  line-height: 1.15;
  word-break: keep-all;
}
.voice-controls[data-state="listening"] .orb-wrapper::before {
  border-color: rgba(244, 63, 94, 0.55);
  background: radial-gradient(circle, rgba(236, 72, 153, 0.28) 0%, transparent 72%);
  animation: ring-ripple 1.4s infinite ease-out;
}
.voice-controls[data-state="listening"] .orb-button {
  background: linear-gradient(135deg, #ef4444 0%, #ec4899 52%, #8b5cf6 100%);
  box-shadow: 0 0 36px rgba(236, 72, 153, 0.68), inset 0 2px 8px rgba(255, 255, 255, 0.42);
  animation: orb-pulse 1.4s infinite ease-in-out;
}
.voice-controls[data-state="speaking"] .orb-wrapper::before {
  border-color: rgba(16, 185, 129, 0.45);
  background: radial-gradient(circle, rgba(16, 185, 129, 0.22) 0%, transparent 72%);
}
.voice-controls[data-state="speaking"] .orb-button {
  background: linear-gradient(135deg, #10b981 0%, #06b6d4 52%, #3b82f6 100%);
  box-shadow: 0 0 32px rgba(16, 185, 129, 0.58), inset 0 2px 8px rgba(255, 255, 255, 0.42);
}
@keyframes orb-pulse {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.06); }
}
@keyframes ring-ripple {
  0% { transform: scale(0.92); opacity: 0.9; }
  100% { transform: scale(1.16); opacity: 0.15; }
}
.voice-transcript {
  margin: 0;
  min-height: 1.35rem;
  font-size: 0.95rem;
  font-weight: 600;
  color: #f1f5f9;
  max-width: 92%;
  overflow-wrap: anywhere;
}
#voice-status {
  margin: 0;
  font-size: 0.85rem;
  color: #cbd5e1;
  line-height: 1.45;
  max-width: 95%;
}
.sub-controls {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  align-items: center;
  gap: 0.5rem;
  width: 100%;
  padding-top: 0.55rem;
  border-top: 1px solid rgba(255, 255, 255, 0.1);
}
.sub-controls button {
  min-height: 2.5rem;
  padding: 0.45rem 0.9rem;
  border: 1px solid rgba(255, 255, 255, 0.18);
  border-radius: 9999px;
  background: rgba(255, 255, 255, 0.08);
  color: #f8fafc;
  font-size: 0.82rem;
  font-weight: 500;
  cursor: pointer;
  transition: background 0.15s ease, border-color 0.15s ease, transform 0.15s ease;
}
.sub-controls button:hover:not(:disabled) {
  background: rgba(255, 255, 255, 0.16);
  border-color: rgba(255, 255, 255, 0.32);
  transform: translateY(-1px);
}
.sub-controls button:disabled {
  cursor: not-allowed;
  opacity: 0.42;
}
@media (max-width: 420px) {
  .voice-controls {
    padding: 1.25rem 0.85rem 1rem;
    border-radius: 1.3rem;
  }
  .orb-wrapper::before {
    width: 8rem;
    height: 8rem;
  }
  .orb-button {
    width: 6.85rem;
    height: 6.85rem;
  }
  .sub-controls button {
    font-size: 0.78rem;
    padding: 0.4rem 0.72rem;
  }
}
""",
    js="""
export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  const root = parentElement.querySelector(".voice-controls");
  if (!root) return;

  const SILENCE_DELAY_MS = 1400;
  const startButton = root.querySelector("#listen-start");
  const orbLabel = root.querySelector(".orb-label");
  const stopButton = root.querySelector("#listen-stop");
  const replayButton = root.querySelector("#speak-again");
  const cancelButton = root.querySelector("#speak-stop");
  const transcriptEl = root.querySelector("#voice-transcript");
  const status = root.querySelector("#voice-status");
  const state = root.__speakerState || (root.__speakerState = {
    recognition: null,
    silenceTimer: null,
    collectedText: "",
    hasSubmitted: false,
    manualStop: false,
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

  function setTranscript(text) {
    if (transcriptEl) transcriptEl.textContent = text;
  }

  function updateVisualState() {
    if (state.recognition) {
      root.setAttribute("data-state", "listening");
      if (orbLabel) orbLabel.textContent = "듣고 있어요...";
    } else if (state.utterance) {
      root.setAttribute("data-state", "speaking");
      if (orbLabel) orbLabel.textContent = "음성 입력 시작";
    } else {
      root.setAttribute("data-state", "idle");
      if (orbLabel) orbLabel.textContent = "음성 입력 시작";
    }
    startButton.disabled = !Recognition || Boolean(state.recognition);
    stopButton.disabled = !state.recognition;
    replayButton.disabled = !synthesis || !data.speechText || !data.voiceEnabled;
    cancelButton.disabled = !state.utterance;
  }

  function clearSilenceTimer() {
    if (state.silenceTimer) {
      clearTimeout(state.silenceTimer);
      state.silenceTimer = null;
    }
  }

  function finishListening({ shouldSubmit, emptyMessage }) {
    clearSilenceTimer();
    const activeRecognition = state.recognition;
    state.recognition = null;
    if (activeRecognition) {
      try {
        activeRecognition.stop();
      } catch (_) {}
    }
    updateVisualState();

    if (!shouldSubmit || state.hasSubmitted) return;
    const finalTranscript = (state.collectedText || "").trim();
    if (!finalTranscript) {
      setTranscript("");
      setStatus(
        emptyMessage ||
          "1.4초 동안 음성이 들리지 않아 음성 인식을 종료했어요. 중앙 버튼을 눌러 다시 말씀해 주세요."
      );
      return;
    }
    state.hasSubmitted = true;
    setStatus("음성을 인식했어요. 답변을 준비할게요.");
    setTriggerValue("recognized", {
      id: `${Date.now()}-${Math.random()}`,
      text: finalTranscript
    });
  }

  function resetSilenceTimer() {
    clearSilenceTimer();
    state.silenceTimer = setTimeout(() => {
      if (!state.recognition) return;
      finishListening({
        shouldSubmit: true,
        emptyMessage:
          "1.4초 동안 음성이 들리지 않아 음성 인식을 종료했어요. 중앙 버튼을 눌러 다시 말씀해 주세요."
      });
    }, SILENCE_DELAY_MS);
  }

  function cancelSpeech(message) {
    state.utterance = null;
    if (synthesis) synthesis.cancel();
    updateVisualState();
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
      updateVisualState();
      setStatus("답변을 읽고 있어요.");
    };
    utterance.onend = () => {
      if (state.utterance !== utterance) return;
      state.utterance = null;
      updateVisualState();
      setStatus("답변 읽기가 끝났어요.");
    };
    utterance.onerror = (event) => {
      if (state.utterance !== utterance) return;
      state.utterance = null;
      updateVisualState();
      setStatus(event.error === "canceled" || event.error === "interrupted"
        ? "답변 읽기를 중지했어요."
        : "자동 재생에 실패했어요. 다시 듣기 버튼을 눌러 보세요.");
    };
    state.utterance = utterance;
    updateVisualState();
    setStatus("답변을 자동으로 읽으려고 해요. 재생되지 않으면 다시 듣기를 눌러 주세요.");
    synthesis.speak(utterance);
  }

  if (state.stopId !== data.stopSpeechId) {
    state.stopId = data.stopSpeechId;
    cancelSpeech("");
  }
  if (state.voiceEnabled && !data.voiceEnabled) {
    cancelSpeech("음성 답변이 꺼졌어요.");
  }
  state.voiceEnabled = data.voiceEnabled;
  updateVisualState();

  if (!Recognition && !status.textContent) {
    setStatus("이 브라우저는 음성 입력을 지원하지 않아요. 텍스트 입력을 이용해 주세요.");
  } else if (!synthesis && !status.textContent) {
    setStatus("이 환경은 음성 읽기를 지원하지 않아요. 답변은 텍스트로 확인해 주세요.");
  } else if (!status.textContent) {
    setStatus("중앙의 둥근 버튼을 누르면 음성 입력이 시작돼요. 1.4초 동안 말이 멈추면 자동으로 전송됩니다.");
  }

  startButton.onclick = () => {
    if (!Recognition) {
      setStatus("이 브라우저는 음성 입력을 지원하지 않아요. 텍스트 입력을 이용해 주세요.");
      return;
    }
    if (state.recognition) return;
    cancelSpeech("");

    const recognition = new Recognition();
    state.recognition = recognition;
    state.collectedText = "";
    state.hasSubmitted = false;
    state.manualStop = false;
    setTranscript("");

    recognition.lang = "ko-KR";
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.onstart = () => {
      if (state.recognition !== recognition) return;
      resetSilenceTimer();
    };
    recognition.onspeechstart = () => {
      if (state.recognition !== recognition) return;
      resetSilenceTimer();
    };
    recognition.onresult = (event) => {
      if (state.hasSubmitted || state.manualStop) return;
      const transcript = Array.from(event.results)
        .map((result) => result[0].transcript)
        .join("")
        .trim();
      state.collectedText = transcript;
      setTranscript(transcript ? `"${transcript}"` : "");
      resetSilenceTimer();
    };
    recognition.onerror = (event) => {
      clearSilenceTimer();
      state.recognition = null;
      state.manualStop = true;
      updateVisualState();
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
      clearSilenceTimer();
      state.recognition = null;
      updateVisualState();
      if (!state.manualStop && !state.hasSubmitted) {
        if ((state.collectedText || "").trim()) {
          finishListening({ shouldSubmit: true });
        } else {
          setStatus("음성을 알아듣지 못했어요. 다시 시도하거나 텍스트로 입력해 주세요.");
        }
      }
    };
    try {
      recognition.start();
      updateVisualState();
      setStatus("듣고 있어요. 1.4초 동안 음성이 들리지 않으면 자동으로 인식을 종료하고 전송합니다.");
      resetSilenceTimer();
    } catch (error) {
      clearSilenceTimer();
      state.recognition = null;
      updateVisualState();
      setStatus(error.name === "NotAllowedError"
        ? "마이크 권한이 없어 음성 입력을 시작할 수 없어요. 권한을 허용하거나 텍스트로 입력해 주세요."
        : "음성 입력을 시작하지 못했어요. 텍스트 입력을 이용해 주세요.");
    }
  };

  stopButton.onclick = () => {
    state.manualStop = true;
    clearSilenceTimer();
    if (state.recognition) {
      try {
        state.recognition.stop();
      } catch (_) {}
      state.recognition = null;
    }
    updateVisualState();
    setStatus("음성 입력을 종료했어요.");
  };
  replayButton.onclick = speakAnswer;
  cancelButton.onclick = () => cancelSpeech("답변 읽기를 중지했어요.");

  if (data.speechId && state.lastSpeechId !== data.speechId) {
    state.lastSpeechId = data.speechId;
    setTranscript("");
    if (data.voiceEnabled) speakAnswer();
  }

  return () => {
    clearSilenceTimer();
    if (state.recognition) {
      try {
        state.recognition.stop();
      } catch (_) {}
    }
    if (synthesis) synthesis.cancel();
  };
}
""",
)


st.set_page_config(page_title="AI 스피커", page_icon=":material/graphic_eq:", layout="centered")
st.html(_PAGE_STYLE)
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
