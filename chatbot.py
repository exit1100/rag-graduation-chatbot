import streamlit as st
from dotenv import load_dotenv
from llm import get_ai_response
import os, time, uuid
from langchain_core.runnables import RunnableConfig
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
from langsmith import Client
from langchain_core.tracers import LangChainTracer
from streamlit_feedback import streamlit_feedback

st.set_page_config(page_title="Graduation Chatbot", page_icon="🤖")
st.title("🤖 중부대학교 졸업 챗봇")
st.caption("중부대학교 학과별 졸업 시 필요한 학점에 대해 알려드립니다.")

def check_if_key_exists(key):
    return key in st.session_state

@st.cache_data(ttl="2h", show_spinner=False)
def get_run_url(run_id):
    time.sleep(1)
    return client.read_run(run_id).url

# API KEY
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "SELF_LEARNING_GPT"

if "query" not in st.session_state:
    st.session_state.query = None

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API KEY", type="password")
    langchain_api_key = st.text_input("LangSmith API KEY (선택사항)", type="password")

    if openai_api_key:
        st.session_state["openai_api_key"] = openai_api_key
        os.environ["OPENAI_API_KEY"] = st.session_state["openai_api_key"]
    
    if langchain_api_key:
        st.session_state["langchain_api_key"] = langchain_api_key

    project_name = st.text_input("LangSmith 프로젝트 (선택사항)", value="SELF_LEARNING_GPT")
    session_id = st.text_input("세션 ID(선택사항)")
    
if not check_if_key_exists("langchain_api_key"):
    st.info(
        "⚠️ [LangSmith API key](https://python.langchain.com/docs/guides/langsmith/walkthrough) 를 추가하면 답변 추적이 가능합니다."
    )
    cfg = RunnableConfig()
    cfg["configurable"] = {"session_id": "asdf1234"}
else:
    langchain_endpoint = "https://api.smith.langchain.com"
    client = Client(
        api_url=langchain_endpoint, api_key=st.session_state["langchain_api_key"]
    )
    ls_tracer = LangChainTracer(project_name=project_name, client=client)
    run_collector = RunCollectorCallbackHandler()
    cfg = RunnableConfig()
    cfg["callbacks"] = [ls_tracer, run_collector]
    if session_id:
        cfg["configurable"] = {"session_id": session_id}
    else:
        cfg["configurable"] = {"session_id": str(uuid.uuid4())}

if not check_if_key_exists("openai_api_key"):
    st.info(
        "⚠️ [OpenAI API key](https://platform.openai.com/docs/guides/authentication) 를 추가해 주세요."
    )

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if user_question := st.chat_input(placeholder="궁금한 내용들을 말씀해주세요!"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.spinner("답변을 생성하는 중입니다"):
        ai_response = get_ai_response(user_question, cfg)
        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response)
            st.session_state.message_list.append({"role": "ai", "content": ai_message})
        if check_if_key_exists("langchain_api_key"):
            wait_for_all_tracers()
            st.session_state.last_run = run_collector.traced_runs[0].id

if st.session_state.get("last_run"):
    run_url = get_run_url(st.session_state.last_run)
    st.sidebar.markdown(f"[LangSmith 추적🛠️]({run_url})")
    feedback = streamlit_feedback(
        feedback_type="thumbs",
        optional_text_label=None,
        key=f"feedback_{st.session_state.last_run}",
    )
    if feedback:
        scores = {"👍": 1, "👎": 0}
        client.create_feedback(
            st.session_state.last_run,
            feedback["type"],
            score=scores[feedback["score"]],
            comment=st.session_state.query,
        )
        st.toast("피드백을 저장하였습니다.!", icon="📝")