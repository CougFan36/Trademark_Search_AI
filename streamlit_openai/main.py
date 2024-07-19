import os
import tempfile
import streamlit as st
from streamlit_chat import message
from rag import ChatDataFrame

st.set_page_config(page_title="AI Trademark Checker")

def display_messages():
    st.subheader("Check to see if your business name is already being used.")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    if st.session_state["user_input"] and\
       len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner("Thinking..."):
            agent_text = st.session_state["assistant"].tm_search(user_text)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))


def page():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["assistant"] = ChatDataFrame()

    st.image("tradmarkai.png", caption="Your friendly AI lawyer.", output_format="auto")

    st.header("Check your trademark")

    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()
    st.text_input("Enter the term you'd like to trademark and press Enter.", key="user_input", on_change=process_input)

if __name__ == "__main__":
    page()
