import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os

from dotenv import load_dotenv
load_dotenv()

search=DuckDuckGoSearchRun(name='search')
st.title("Langchain - chat with search")

g_api_key=os.getenv('GROQ_API_KEY')

if "messages" not in st.session_state:
    st.session_state["messages"]=[{"role":"assistant", "content":"Hi, I am a chatbot who can search web"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt:=st.chat_input(placeholder="schools in Bengaluru"):
    st.session_state.messages.append({"role":"user", "content":prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(groq_api_key=g_api_key, model="llama-3.1-8b-instant", streaming=True)
    tools=[search]

    search_agent=initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_error=True)

    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response=search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role":"assistant", "content":response})
        st.write(response)