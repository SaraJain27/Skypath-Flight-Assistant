
import uuid
import streamlit as st


st.set_page_config(page_title="Skypath", page_icon="✈️", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --sky: #FFFFFF;
    --panel: #FFFFFF;
    --panel-2: #F3F5FA;
    --ink: #10142B;
    --amber: #E0902B;
    --teal: #12967F;
    --slate: #5B6480;
    --line: #E4E8F2;
}

.stApp {
    background: radial-gradient(1200px 800px at 80% -10%, #F3F6FC 0%, var(--sky) 55%);
    color: var(--ink);
    font-family: 'Inter', sans-serif;
}

/* Title + caption */
h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    color: var(--ink) !important;
    padding-bottom: 0.3rem !important;
}
[data-testid="stCaptionContainer"], .stCaption, p:has(> em) {
    color: var(--slate) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* Signature element: runway lights strip under the header ONLY --
   scoped specifically to the caption text, not any generic container,
   so it doesn't bleed onto chat messages or the input box. */
[data-testid="stCaptionContainer"] {
    border-bottom: 2px dashed var(--amber) !important;
    padding-bottom: 1.2rem !important;
    margin-bottom: 1.8rem !important;
}

/* Chat messages -- base style */
[data-testid="stChatMessage"] {
    border-radius: 14px !important;
    padding: 4px 6px !important;
    margin-bottom: 10px !important;
    box-shadow: 0 1px 3px rgba(16, 20, 43, 0.06) !important;
}
[data-testid="stChatMessage"] p {
    color: var(--ink) !important;
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
}
[data-testid="stChatMessage"] a {
    color: var(--teal) !important;
}

/* User messages (odd position: 1st, 3rd, 5th...) -- warm amber tint */
[data-testid="stChatMessage"]:nth-of-type(odd) {
    background: #FBF1E4 !important;
    border: 1px solid #EBD3A8 !important;
}

/* Assistant messages (even position: 2nd, 4th, 6th...) -- cool teal tint */
[data-testid="stChatMessage"]:nth-of-type(even) {
    background: #EAF5F3 !important;
    border: 1px solid #BFE3DC !important;
}

/* Chat input box */
[data-testid="stChatInput"] textarea {
    background: var(--panel-2) !important;
    color: var(--ink) !important;
    border: 1px solid var(--line) !important;
    border-radius: 20px !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--slate) !important;
}
[data-testid="stChatInput"] button {
    background: var(--amber) !important;
    border-radius: 50% !important;
}

/* Spinner text */
[data-testid="stSpinner"] p {
    color: var(--slate) !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
}
</style>
""", unsafe_allow_html=True)

st.title("✈️ Skypath — Flight Assistant")



with st.spinner("Loading Skypath (first run can take up to a minute -- "
                 "downloading the embedding model, building the search index)..."):
    from step8_full_assistant import ask



if "thread_id" not in st.session_state:
    # A unique ID per browser session -- this is what our checkpointer
    # (Step 5) uses to keep each user's conversation memory separate.
    st.session_state.thread_id = "streamlit-" + str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    avatar = "✈️" if msg["role"] == "assistant" else "🧳"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])


user_input = st.chat_input("Ask about a flight, or an airline policy...")

if user_input:
    # 1. Show the user's own message immediately.
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧳"):
        st.markdown(user_input)

    # 2. Call our EXISTING agent logic -- same ask() function from
    #    step8_full_assistant.py, same thread_id for this whole session,
    #    so memory (Step 5) works correctly across turns in this chat.
    with st.chat_message("assistant", avatar="✈️"):
        with st.spinner("Thinking..."):
            answer = ask(user_input, thread_id=st.session_state.thread_id)
            st.markdown(answer)

    # 3. Save the assistant's reply so it's still shown after the next re-run.
    st.session_state.messages.append({"role": "assistant", "content": answer})
