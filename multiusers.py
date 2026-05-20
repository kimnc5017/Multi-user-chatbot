"""멀티유저/멀티세션 RAG 챗봇 — Supabase user 테이블 + 세션/벡터 저장."""

from __future__ import annotations

import logging
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import bcrypt
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase import Client, create_client

# ---------------------------------------------------------------------------
# Paths & environment
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPO_ROOT / ".env"
LOGO_PATH = REPO_ROOT / "logo.png"
LOG_DIR = REPO_ROOT / "logs"

load_dotenv(dotenv_path=ENV_PATH)

EMBED_BATCH_SIZE = 10
SEARCH_K = 10
CHAT_MODEL = "gpt-4o-mini"
USER_TABLE = "user"

ANSWER_STYLE_SYSTEM = """당신은 친절하고 공손한 AI 어시스턴트입니다.

답변 규칙:
- 반드시 마크다운 헤딩(# ## ###)으로 구조화하세요. 주요 주제는 #, 세부는 ##, 구체 설명은 ###.
- 서술형으로 완전한 문장을 사용하고 존댓말로 작성하세요.
- 구분선(---, ===, ___)은 사용하지 마세요.
- 취소선(~~텍스트~~)은 사용하지 마세요.
- 참조 표시, 각주, 출처 문구, URL 인용 문장은 넣지 마세요.
"""


def _writable_log_dir() -> Path | None:
    """Streamlit Cloud 등 읽기 전용 환경에서는 /tmp 등으로 대체."""
    for candidate in (LOG_DIR, Path(tempfile.gettempdir()) / "multiusers_logs"):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError:
            continue
    return None


def _setup_logging() -> logging.Logger:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    log_dir = _writable_log_dir()
    if log_dir is not None:
        try:
            log_path = log_dir / f"multiusers_{datetime.now().strftime('%Y%m%d')}.log"
            fh = logging.FileHandler(log_path, encoding="utf-8")
            fh.setLevel(logging.WARNING)
            fh.setFormatter(fmt)
            root.addHandler(fh)
        except OSError:
            pass

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    for name in ("httpx", "httpcore", "urllib3", "openai", "langchain", "langchain_openai"):
        logging.getLogger(name).setLevel(logging.WARNING)

    return logging.getLogger("multiusers")


logger = _setup_logging()


# ---------------------------------------------------------------------------
# Secrets & helpers
# ---------------------------------------------------------------------------
def _get_secret(key: str) -> str:
    """Streamlit Cloud: st.secrets 우선, 없으면 .env / os.getenv."""
    try:
        value = st.secrets.get(key)
        if value:
            return str(value).strip()
    except Exception:
        pass
    return os.getenv(key, "").strip()


def remove_separators(text: str) -> str:
    out = re.sub(r"~~([^~]*)~~", r"\1", text)
    out = re.sub(r"(?m)^\s*-{3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*={3,}\s*$", "", out)
    out = re.sub(r"(?m)^\s*_{3,}\s*$", "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def get_env_status() -> dict[str, bool]:
    return {
        "OPENAI_API_KEY": bool(_get_secret("OPENAI_API_KEY")),
        "SUPABASE_URL": bool(_get_secret("SUPABASE_URL")),
        "SUPABASE_ANON_KEY": bool(_get_secret("SUPABASE_ANON_KEY")),
    }


def missing_env_message() -> str | None:
    status = get_env_status()
    missing = [k for k, ok in status.items() if not ok]
    if not missing:
        return None
    return (
        "# 환경 변수 안내\n\n"
        "다음 키가 설정되어 있지 않습니다:\n\n"
        + "\n".join(f"- `{k}`" for k in missing)
        + "\n\n로컬: `.env` · Streamlit Cloud: `st.secrets`에 설정하세요.\n\n"
        f"로컬 `.env` 경로: `{ENV_PATH}`"
    )


def get_supabase() -> Client | None:
    url = _get_secret("SUPABASE_URL")
    key = _get_secret("SUPABASE_ANON_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


def get_openai_key() -> str:
    return _get_secret("OPENAI_API_KEY")


def get_llm() -> ChatOpenAI:
    key = get_openai_key()
    if not key:
        raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
    return ChatOpenAI(model=CHAT_MODEL, temperature=0.7, api_key=key)


def get_embeddings() -> OpenAIEmbeddings:
    key = get_openai_key()
    if not key:
        raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
    return OpenAIEmbeddings(api_key=key)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# User auth (user 테이블 — Supabase Auth 미사용)
# ---------------------------------------------------------------------------
def register_user(client: Client, login_id: str, password: str) -> tuple[bool, str]:
    login_id = login_id.strip()
    if not login_id:
        return False, "아이디를 입력하세요."
    if len(password) < 4:
        return False, "비밀번호는 4자 이상이어야 합니다."

    existing = (
        client.table(USER_TABLE)
        .select("id")
        .eq("login_id", login_id)
        .limit(1)
        .execute()
    )
    if existing.data:
        return False, "이미 사용 중인 아이디입니다."

    row = {"login_id": login_id, "password_hash": hash_password(password)}
    resp = client.table(USER_TABLE).insert(row).execute()
    if not resp.data:
        return False, "회원가입에 실패했습니다."
    return True, "회원가입이 완료되었습니다. 로그인하세요."


def login_user(client: Client, login_id: str, password: str) -> tuple[dict[str, Any] | None, str]:
    login_id = login_id.strip()
    if not login_id or not password:
        return None, "아이디와 비밀번호를 입력하세요."

    resp = (
        client.table(USER_TABLE)
        .select("id, login_id, password_hash")
        .eq("login_id", login_id)
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    if not rows:
        return None, "아이디 또는 비밀번호가 올바르지 않습니다."

    user = rows[0]
    if not verify_password(password, user["password_hash"]):
        return None, "아이디 또는 비밀번호가 올바르지 않습니다."

    return {"id": str(user["id"]), "login_id": user["login_id"]}, "로그인 성공"


def get_current_user() -> dict[str, Any] | None:
    return st.session_state.get("current_user")


def get_current_user_id() -> str | None:
    user = get_current_user()
    return str(user["id"]) if user else None


def logout_user() -> None:
    st.session_state.current_user = None
    clear_screen()


# ---------------------------------------------------------------------------
# Supabase — sessions & messages (user_id 필터)
# ---------------------------------------------------------------------------
def fetch_sessions(client: Client, user_id: str) -> list[dict[str, Any]]:
    resp = (
        client.table("chat_sessions")
        .select("id, title, processed_files, created_at, updated_at")
        .eq("user_id", user_id)
        .order("updated_at", desc=True)
        .execute()
    )
    return resp.data or []


def fetch_messages(client: Client, user_id: str, session_id: str) -> list[dict[str, Any]]:
    resp = (
        client.table("chat_messages")
        .select("role, content, sequence_num")
        .eq("user_id", user_id)
        .eq("session_id", session_id)
        .order("sequence_num")
        .execute()
    )
    return resp.data or []


def generate_session_title(llm: ChatOpenAI, user_q: str, assistant_a: str) -> str:
    prompt = (
        "다음 첫 질문과 답변을 바탕으로 대화 세션 제목을 한국어로 한 줄(20자 내외)로만 작성하세요.\n"
        "따옴표, 설명, 부가 문구 없이 제목만 출력하세요.\n\n"
        f"[질문]\n{user_q[:500]}\n\n[답변]\n{assistant_a[:800]}"
    )
    try:
        out = llm.invoke([HumanMessage(content=prompt)])
        title = str(getattr(out, "content", out) or "").strip()
        title = title.strip("\"'")
        return title[:80] if title else "새 세션"
    except Exception as exc:  # noqa: BLE001
        logger.warning("세션 제목 생성 실패: %s", exc)
        return user_q[:40] or "새 세션"


def _first_qa_pair(messages: list[dict[str, str]]) -> tuple[str, str] | None:
    user_q = ""
    for m in messages:
        if m["role"] == "user" and not user_q:
            user_q = m["content"]
        elif m["role"] == "assistant" and user_q:
            return user_q, m["content"]
    return None


def insert_session(
    client: Client,
    user_id: str,
    *,
    title: str,
    processed_files: list[str],
) -> str:
    row = {
        "user_id": user_id,
        "title": title,
        "processed_files": processed_files,
    }
    resp = client.table("chat_sessions").insert(row).execute()
    if not resp.data:
        raise RuntimeError("세션 생성에 실패했습니다.")
    return str(resp.data[0]["id"])


def save_messages(
    client: Client,
    user_id: str,
    session_id: str,
    messages: list[dict[str, str]],
) -> None:
    (
        client.table("chat_messages")
        .delete()
        .eq("user_id", user_id)
        .eq("session_id", session_id)
        .execute()
    )
    rows = [
        {
            "user_id": user_id,
            "session_id": session_id,
            "role": m["role"],
            "content": m["content"],
            "sequence_num": i,
        }
        for i, m in enumerate(messages)
    ]
    if rows:
        client.table("chat_messages").insert(rows).execute()


def update_session_meta(
    client: Client,
    user_id: str,
    session_id: str,
    *,
    title: str | None = None,
    processed_files: list[str] | None = None,
) -> None:
    payload: dict[str, Any] = {}
    if title is not None:
        payload["title"] = title
    if processed_files is not None:
        payload["processed_files"] = processed_files
    if payload:
        (
            client.table("chat_sessions")
            .update(payload)
            .eq("id", session_id)
            .eq("user_id", user_id)
            .execute()
        )


def delete_session_db(client: Client, user_id: str, session_id: str) -> None:
    (
        client.table("chat_sessions")
        .delete()
        .eq("id", session_id)
        .eq("user_id", user_id)
        .execute()
    )


def copy_vectors_to_session(
    client: Client,
    source_session_id: str,
    target_session_id: str,
) -> None:
    if source_session_id == target_session_id:
        return
    resp = (
        client.table("vector_documents")
        .select("file_name, content, embedding, metadata")
        .eq("session_id", source_session_id)
        .execute()
    )
    rows = resp.data or []
    if not rows:
        return
    for i in range(0, len(rows), EMBED_BATCH_SIZE):
        batch = rows[i : i + EMBED_BATCH_SIZE]
        inserts = [{**r, "session_id": target_session_id} for r in batch]
        client.table("vector_documents").insert(inserts).execute()


# ---------------------------------------------------------------------------
# Supabase — vectors
# ---------------------------------------------------------------------------
def embed_and_store_pdfs(
    client: Client,
    session_id: str,
    uploaded_files: list[Any],
) -> list[str]:
    api_key = get_openai_key()
    if not api_key:
        raise ValueError("PDF 임베딩에 OPENAI_API_KEY가 필요합니다.")

    embeddings = OpenAIEmbeddings(api_key=api_key)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    processed_names: list[str] = []

    for uf in uploaded_files:
        suffix = Path(uf.name).suffix.lower() or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uf.getvalue())
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        if not docs:
            continue

        file_name = Path(uf.name).name
        processed_names.append(file_name)
        splits = splitter.split_documents(docs)

        for i in range(0, len(splits), EMBED_BATCH_SIZE):
            batch = splits[i : i + EMBED_BATCH_SIZE]
            texts = [d.page_content for d in batch]
            vectors = embeddings.embed_documents(texts)
            rows = []
            for doc, vec in zip(batch, vectors, strict=True):
                rows.append(
                    {
                        "session_id": session_id,
                        "file_name": file_name,
                        "content": doc.page_content,
                        "embedding": vec,
                        "metadata": {
                            "source": doc.metadata.get("source", ""),
                            "page": doc.metadata.get("page"),
                        },
                    }
                )
            client.table("vector_documents").insert(rows).execute()

    return processed_names


def fetch_vector_file_names(client: Client, session_id: str | None) -> list[str]:
    if not session_id:
        return []
    resp = (
        client.table("vector_documents")
        .select("file_name")
        .eq("session_id", session_id)
        .execute()
    )
    names = {row["file_name"] for row in (resp.data or []) if row.get("file_name")}
    return sorted(names)


def session_has_vectors(client: Client, session_id: str) -> bool:
    resp = (
        client.table("vector_documents")
        .select("id")
        .eq("session_id", session_id)
        .limit(1)
        .execute()
    )
    return bool(resp.data)


def retrieve_documents(
    client: Client,
    session_id: str,
    query: str,
    k: int = SEARCH_K,
) -> list[Document]:
    embeddings = get_embeddings()
    query_vec = embeddings.embed_query(query)

    try:
        resp = client.rpc(
            "match_vector_documents",
            {
                "query_embedding": query_vec,
                "match_count": k,
                "filter_session_id": session_id,
            },
        ).execute()
        rows = resp.data or []
        return [
            Document(
                page_content=row["content"],
                metadata={
                    "file_name": row.get("file_name"),
                    "similarity": row.get("similarity"),
                    **(row.get("metadata") or {}),
                },
            )
            for row in rows
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("RPC 검색 실패, 대체 필터 사용: %s", exc)
        resp = (
            client.table("vector_documents")
            .select("content, file_name, metadata, embedding")
            .eq("session_id", session_id)
            .execute()
        )
        docs: list[Document] = []
        for row in resp.data or []:
            docs.append(
                Document(
                    page_content=row["content"],
                    metadata={"file_name": row.get("file_name"), **(row.get("metadata") or {})},
                )
            )
        return docs[:k]


# ---------------------------------------------------------------------------
# RAG / follow-up / streaming
# ---------------------------------------------------------------------------
def _format_memory_block(messages: list[dict[str, str]], max_items: int = 50) -> str:
    tail = messages[-max_items:] if len(messages) > max_items else messages
    lines: list[str] = []
    for m in tail:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        prefix = "사용자" if role == "user" else "어시스턴트"
        lines.append(f"{prefix}: {content}")
    return "\n".join(lines)


def _build_rag_messages(
    question: str,
    context: str,
    memory_text: str,
) -> list[SystemMessage | HumanMessage]:
    sys = f"""{ANSWER_STYLE_SYSTEM}

아래 [대화 맥락]과 [참고 문서]를 활용해 답하세요. 참고 문서에 없는 내용은 추측하지 말고 한계를 밝히세요.
[대화 맥락]
{memory_text or "(없음)"}

[참고 문서]
{context}
"""
    return [SystemMessage(content=sys), HumanMessage(content=question)]


def _generate_followup_section(llm: ChatOpenAI, user_q: str, answer: str) -> str:
    trimmed = answer[:8000]
    prompt = (
        "다음 사용자 질문과 답변을 바탕으로, 이어서 물어볼 만한 후속 질문을 한국어로 정확히 3개만 작성하세요.\n"
        "형식:\n1. ...\n2. ...\n3. ...\n"
        "설명 문장이나 다른 텍스트는 출력하지 마세요.\n\n"
        f"[사용자 질문]\n{user_q}\n\n[답변]\n{trimmed}"
    )
    try:
        out = llm.invoke([HumanMessage(content=prompt)])
        raw = remove_separators(str(getattr(out, "content", out) or ""))
    except Exception as exc:  # noqa: BLE001
        logger.warning("후속 질문 생성 실패: %s", exc)
        return ""
    if not raw.strip():
        return ""
    return f"\n\n### 💡 다음에 물어볼 수 있는 질문들\n\n{raw.strip()}\n"


def _stream_llm(
    llm: ChatOpenAI,
    messages: list[SystemMessage | HumanMessage],
    placeholder: Any,
) -> str:
    acc = ""
    for chunk in llm.stream(messages):
        piece = getattr(chunk, "content", "") or ""
        if piece:
            acc += piece
            placeholder.markdown(remove_separators(acc) + "▌")
    return remove_separators(acc)


# ---------------------------------------------------------------------------
# Session state & UI actions
# ---------------------------------------------------------------------------
def _init_session() -> None:
    defaults: dict[str, Any] = {
        "current_user": None,
        "chat_history": [],
        "conversation_memory": [],
        "current_session_id": None,
        "processed_names": [],
        "sessions_cache": [],
        "sidebar_selected_id": None,
        "last_loaded_session_id": None,
        "auth_mode": "login",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def apply_session_to_ui(
    client: Client,
    user_id: str,
    session_id: str,
    sessions: list[dict[str, Any]],
) -> None:
    messages = fetch_messages(client, user_id, session_id)
    st.session_state.chat_history = [
        {"role": m["role"], "content": m["content"]} for m in messages
    ]
    st.session_state.conversation_memory = list(st.session_state.chat_history)
    st.session_state.current_session_id = session_id
    st.session_state.last_loaded_session_id = session_id

    meta = next((s for s in sessions if s["id"] == session_id), None)
    if meta and meta.get("processed_files"):
        st.session_state.processed_names = list(meta["processed_files"])
    elif session_has_vectors(client, session_id):
        st.session_state.processed_names = fetch_vector_file_names(client, session_id)
    else:
        st.session_state.processed_names = []


def auto_save_session(client: Client, user_id: str) -> None:
    if not st.session_state.chat_history and not st.session_state.processed_names:
        return

    llm = get_llm()
    session_id = st.session_state.current_session_id
    title = "새 세션"
    pair = _first_qa_pair(st.session_state.chat_history)
    if pair:
        title = generate_session_title(llm, pair[0], pair[1])

    files = list(st.session_state.processed_names)

    if session_id:
        update_session_meta(client, user_id, session_id, title=title, processed_files=files)
        save_messages(client, user_id, session_id, st.session_state.chat_history)
    else:
        session_id = insert_session(
            client, user_id, title=title, processed_files=files
        )
        st.session_state.current_session_id = session_id
        save_messages(client, user_id, session_id, st.session_state.chat_history)

    st.session_state.sessions_cache = fetch_sessions(client, user_id)


def manual_save_new_session(client: Client, user_id: str) -> str | None:
    if not st.session_state.chat_history:
        st.warning("저장할 대화가 없습니다.")
        return None

    llm = get_llm()
    title = "새 세션"
    pair = _first_qa_pair(st.session_state.chat_history)
    if pair:
        title = generate_session_title(llm, pair[0], pair[1])

    old_id = st.session_state.current_session_id
    new_id = insert_session(
        client,
        user_id,
        title=title,
        processed_files=list(st.session_state.processed_names),
    )
    save_messages(client, user_id, new_id, st.session_state.chat_history)

    if old_id:
        copy_vectors_to_session(client, old_id, new_id)

    st.session_state.current_session_id = new_id
    st.session_state.last_loaded_session_id = new_id
    st.session_state.sessions_cache = fetch_sessions(client, user_id)
    return new_id


def clear_screen() -> None:
    st.session_state.chat_history = []
    st.session_state.conversation_memory = []
    st.session_state.current_session_id = None
    st.session_state.processed_names = []
    st.session_state.last_loaded_session_id = None


def ensure_working_session(client: Client, user_id: str) -> str:
    sid = st.session_state.current_session_id
    if sid:
        return sid
    sid = insert_session(client, user_id, title="새 세션", processed_files=[])
    st.session_state.current_session_id = sid
    st.session_state.sessions_cache = fetch_sessions(client, user_id)
    return sid


# ---------------------------------------------------------------------------
# Auth UI
# ---------------------------------------------------------------------------
def render_auth_screen(client: Client, env_msg: str | None) -> None:
    st.markdown("## 로그인 / 회원가입")
    st.caption("Supabase `user` 테이블 기반 계정입니다. (Supabase Auth 미사용)")

    if env_msg:
        st.markdown(env_msg)
        return

    tab_login, tab_register = st.tabs(["로그인", "회원가입"])

    with tab_login:
        login_id = st.text_input("아이디", key="login_id_input")
        password = st.text_input("비밀번호", type="password", key="login_pw_input")
        if st.button("로그인", use_container_width=True):
            user, msg = login_user(client, login_id, password)
            if user:
                st.session_state.current_user = user
                clear_screen()
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    with tab_register:
        reg_id = st.text_input("아이디", key="reg_id_input")
        reg_pw = st.text_input("비밀번호", type="password", key="reg_pw_input")
        reg_pw2 = st.text_input("비밀번호 확인", type="password", key="reg_pw2_input")
        if st.button("회원가입", use_container_width=True):
            if reg_pw != reg_pw2:
                st.error("비밀번호가 일치하지 않습니다.")
            else:
                ok, msg = register_user(client, reg_id, reg_pw)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="재정경제부 RAG 챗봇",
        page_icon="📚",
        layout="wide",
    )
    _init_session()

    st.markdown(
        """
<style>
h1 { color: #ff69b4 !important; font-size: 1.4rem !important; }
h2 { color: #ffd700 !important; font-size: 1.2rem !important; }
h3 { color: #1f77b4 !important; font-size: 1.1rem !important; }
motion div.stButton > button:first-child {
  background-color: #ff69b4;
  color: #ffffff;
}
motion div.stButton > button {
  background-color: #ff69b4;
  color: #ffffff;
}
</style>
""".replace("motion ", ""),
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1, 4, 1])
    with c1:
        if LOGO_PATH.is_file():
            st.image(str(LOGO_PATH), width=180)
        else:
            st.markdown("### 📚")
    with c2:
        st.markdown(
            """
<h1 style="text-align:center; margin:0;">
  <span style="color:#1f77b4;">재정경제부</span>
  <span style="color:#ff8c00;">RAG 챗봇</span>
</h1>
""",
            unsafe_allow_html=True,
        )
    with c3:
        st.empty()

    env_msg = missing_env_message()
    client = get_supabase()

    user = get_current_user()
    user_id = get_current_user_id()

    if not user:
        if not client and not env_msg:
            st.markdown(
                "# 안내\n\nSupabase URL과 ANON KEY를 설정해 주세요."
            )
            return
        if client:
            render_auth_screen(client, env_msg)
        else:
            st.markdown(env_msg or "# 안내\n\nSupabase 연결 정보를 확인하세요.")
        return

    with st.sidebar:
        st.markdown(f"**로그인:** `{user['login_id']}`")
        if st.button("로그아웃", use_container_width=True):
            logout_user()
            st.rerun()

        st.markdown("---")
        st.markdown("**모델**")
        st.text(CHAT_MODEL)

        if env_msg:
            st.error("환경 변수가 누락되었습니다.")
            for k, ok in get_env_status().items():
                st.text(f"{k}: {'✓' if ok else '✗'}")

        st.markdown("---")
        st.markdown("**세션 관리**")

        sessions: list[dict[str, Any]] = []
        if client and user_id and not env_msg:
            try:
                sessions = fetch_sessions(client, user_id)
                st.session_state.sessions_cache = sessions
            except Exception as exc:  # noqa: BLE001
                st.error(f"세션 목록 조회 실패: {exc}")

        session_ids = [s["id"] for s in sessions]
        id_to_title = {s["id"]: s.get("title") or "제목 없음" for s in sessions}

        if session_ids:
            default_idx = 0
            cur = st.session_state.current_session_id
            if cur in session_ids:
                default_idx = session_ids.index(cur)

            selected_id = st.selectbox(
                "세션 선택",
                options=session_ids,
                index=default_idx,
                format_func=lambda sid: id_to_title.get(sid, sid),
                key="session_selectbox",
            )

            if selected_id != st.session_state.get("last_loaded_session_id"):
                if client and user_id:
                    try:
                        apply_session_to_ui(client, user_id, selected_id, sessions)
                        st.session_state.sidebar_selected_id = selected_id
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"세션 자동 로드 실패: {exc}")
        else:
            selected_id = None
            st.info("저장된 세션이 없습니다.")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("세션저장", use_container_width=True):
                if client and user_id and not env_msg:
                    try:
                        new_id = manual_save_new_session(client, user_id)
                        if new_id:
                            st.success("새 세션이 저장되었습니다.")
                            st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"세션 저장 실패: {exc}")
                else:
                    st.warning("Supabase 설정을 확인하세요.")

            if st.button("세션로드", use_container_width=True):
                if client and user_id and selected_id and not env_msg:
                    try:
                        apply_session_to_ui(client, user_id, selected_id, sessions)
                        st.success("세션을 불러왔습니다.")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"세션 로드 실패: {exc}")
                else:
                    st.warning("불러올 세션을 선택하세요.")

        with col_b:
            if st.button("세션삭제", use_container_width=True):
                sid = st.session_state.current_session_id or selected_id
                if client and user_id and sid and not env_msg:
                    try:
                        delete_session_db(client, user_id, sid)
                        if st.session_state.current_session_id == sid:
                            clear_screen()
                        st.session_state.sessions_cache = fetch_sessions(client, user_id)
                        st.success("세션이 삭제되었습니다.")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"세션 삭제 실패: {exc}")
                else:
                    st.warning("삭제할 세션을 선택하세요.")

            if st.button("화면초기화", use_container_width=True):
                clear_screen()
                st.rerun()

        if st.button("vectordb", use_container_width=True):
            sid = st.session_state.current_session_id
            if client and sid and not env_msg:
                names = fetch_vector_file_names(client, sid)
                if names:
                    st.markdown("**Vector DB 파일 목록**")
                    for n in names:
                        st.text(f"- {n}")
                else:
                    st.info("현재 세션에 저장된 벡터 문서가 없습니다.")
            else:
                st.info("활성 세션이 없거나 Supabase에 연결되지 않았습니다.")

        st.markdown("---")
        uploads = st.file_uploader(
            "PDF 파일 업로드",
            type=["pdf"],
            accept_multiple_files=True,
        )
        if st.button("파일 처리하기"):
            if env_msg:
                st.warning(env_msg)
            elif not uploads:
                st.warning("업로드된 PDF가 없습니다.")
            elif client and user_id:
                try:
                    sid = ensure_working_session(client, user_id)
                    names = embed_and_store_pdfs(client, sid, list(uploads))
                    st.session_state.processed_names = list(
                        dict.fromkeys(st.session_state.processed_names + names)
                    )
                    update_session_meta(
                        client,
                        user_id,
                        sid,
                        processed_files=st.session_state.processed_names,
                    )
                    auto_save_session(client, user_id)
                    st.success("PDF 처리 및 세션 자동 저장이 완료되었습니다.")
                except Exception as exc:  # noqa: BLE001
                    logger.warning("PDF 처리 실패: %s", exc)
                    st.error(f"PDF 처리 중 오류: {exc}")
            else:
                st.warning("Supabase 연결을 확인하세요.")

        if st.session_state.processed_names:
            st.markdown("**처리된 파일**")
            for name in st.session_state.processed_names:
                st.text(f"- {name}")

        sid = st.session_state.current_session_id
        st.text(
            f"현재 세션: {id_to_title.get(sid, '(새 화면)') if sid else '(새 화면)'}\n"
            f"대화 수: {len(st.session_state.chat_history)}\n"
            f"벡터 파일 수: {len(st.session_state.processed_names)}"
        )

    if env_msg:
        st.markdown(env_msg)
        return

    if not client:
        st.markdown(
            "# 안내\n\nSupabase URL과 ANON KEY를 설정해 주세요."
        )
        return

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(remove_separators(msg["content"]))

    user_input = st.chat_input("질문을 입력하세요")
    if not user_input:
        return

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.conversation_memory.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(remove_separators(user_input))

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_answer = ""

        try:
            llm = get_llm()
            sid = st.session_state.current_session_id
            use_rag = bool(sid and session_has_vectors(client, sid))

            if use_rag and sid:
                mem_txt = _format_memory_block(st.session_state.conversation_memory[:-1])
                docs = retrieve_documents(client, sid, user_input)
                context = "\n\n".join(d.page_content for d in docs)
                messages = _build_rag_messages(user_input, context, mem_txt)
                full_answer = _stream_llm(llm, messages, placeholder)
            else:
                mem_txt = _format_memory_block(st.session_state.conversation_memory[:-1])
                sys = f"{ANSWER_STYLE_SYSTEM}\n\n[대화 맥락]\n{mem_txt or '(없음)'}"
                msgs = [SystemMessage(content=sys), HumanMessage(content=user_input)]
                full_answer = _stream_llm(llm, msgs, placeholder)

            placeholder.markdown(full_answer)

            if full_answer and not full_answer.lstrip().startswith("# 오류"):
                follow = _generate_followup_section(llm, user_input, full_answer)
                if follow:
                    full_answer += follow
                    placeholder.markdown(remove_separators(full_answer))

        except Exception as exc:  # noqa: BLE001
            logger.warning("답변 생성 실패: %s", exc)
            full_answer = f"# 오류\n\n요청 처리 중 문제가 발생했습니다.\n\n`{exc}`"
            placeholder.markdown(remove_separators(full_answer))

    st.session_state.chat_history.append({"role": "assistant", "content": full_answer})
    st.session_state.conversation_memory.append({"role": "assistant", "content": full_answer})
    if len(st.session_state.conversation_memory) > 50:
        st.session_state.conversation_memory = st.session_state.conversation_memory[-50:]

    try:
        if user_id:
            if not st.session_state.current_session_id:
                ensure_working_session(client, user_id)
            auto_save_session(client, user_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("자동 저장 실패: %s", exc)


if __name__ == "__main__":
    main()
