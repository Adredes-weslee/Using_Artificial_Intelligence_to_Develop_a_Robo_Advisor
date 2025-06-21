"""Streamlit Page: A simple financial chatbot."""
import streamlit as st

st.set_page_config(page_title="Financial Chatbot", page_icon="💬")
st.title("💬 Financial Q&A Chatbot")
st.markdown("Ask a question about common financial concepts.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is a Sharpe Ratio?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Basic keyword-based response logic
    response = "I am a simple demo bot. Here are some concepts I know:\n" \
               "- **Risk Tolerance**: An investor's willingness to accept risk.\n" \
               "- **Sharpe Ratio**: A measure of risk-adjusted return.\n" \
               "- **DQN**: Deep Q-Network, a type of reinforcement learning model."
    if "sharpe" in prompt.lower():
        response = "The Sharpe Ratio measures the performance of an investment compared to a risk-free asset, after adjusting for its risk. A higher Sharpe Ratio is generally better."
    elif "risk" in prompt.lower():
        response = "Risk tolerance is the degree of variability in investment returns that an investor is willing to withstand. Our profiler helps estimate this."

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
