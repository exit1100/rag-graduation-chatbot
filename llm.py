from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_upstage import UpstageEmbeddings, ChatUpstage 
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from config import answer_examples
import os

store = {}
llm_model = 'OpenAI'

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever


def get_llm():
    if llm_model == 'OpenAI':
        return ChatOpenAI(model='gpt-4o')
    elif llm_model == 'Upstage':
        upstage_api_key = os.getenv("UPSTAGE_API_KEY")
        print(upstage_api_key)
        return ChatUpstage(api_key=upstage_api_key)
    else:
        raise ValueError("Invalid provider selected.")


def get_retriever():
    if llm_model == 'OpenAI':
        embedding = OpenAIEmbeddings(model='text-embedding-3-large')
        index_name = 'joongbu-graduation'
    elif llm_model == 'Upstage':
        embedding = UpstageEmbeddings(model='solar-embedding-1-large')
        index_name = 'joongbu-graduation-upstage'
    else:
        raise ValueError("지원하지 않는 모델 소스입니다. 'OpenAI' 또는 'Upstage' 중에서 선택하세요.")

    database = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
    retriever = database.as_retriever(search_kwargs={'k': 4})
    return retriever


def get_dictionary_chain():
    if llm_model == 'OpenAI':
        dictionary = ["CCIT -> 융합전공"]
    elif llm_model == 'Upstage':
        dictionary = []
    
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요.
        사전 : {dictionary}
                                            
        질문 : {{question}} 
    """)
    dictionary_chain = prompt | llm | StrOutputParser()
    return dictionary_chain


def few_shot():
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )
    return few_shot_prompt


def get_rag_chain():
    llm = get_llm()

    system_prompt = (
        "당신은 중부대학교 졸업 기준에 대해 깊이 있는 지식을 가지고 있는 전문가입니다. "
        "답변을 제공할 때는 중부대학교 교육과정 및 졸업기준 가이드(??페이지)에 따르면 이라고 시작하면서 참고 페이지를 제시해주세요. "
        "아래에 제공된 문서를 활용해서 답변해주시고, 답변을 알 수 없다면 모른다고 답변해주세요. "
        "답변은 다섯 문장 이내로 간결하고 가독성이 있도록 유지하세요."
        "입학 연도(학번)와 학과별로 졸업 요건과 이수 학점 기준이 다르니 주의해주세요. "
        "\n\n"
        "{context}"
    )
    few_shot_prompt = few_shot()
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = get_history_retriever()
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer')
    return conversational_rag_chain


def get_ai_response(user_message, cfg, provider):
    global llm_model
    llm_model = provider
    dictionary_chain = get_dictionary_chain()
    rag_chain = get_rag_chain()
    graduation_chain = {"input" : dictionary_chain} | rag_chain
    ai_response = graduation_chain.stream(
        {
            "question": user_message
        },
        config=cfg,
    )
    return ai_response
