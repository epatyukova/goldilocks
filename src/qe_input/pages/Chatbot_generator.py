import streamlit as st
from utils import create_client, generate_llm_response, create_task

st.title("Generate QE input with an LLM Agent")

groq_api_key=None
openai_api_key=None
gemini_api_key=None

if 'all_info' not in st.session_state.keys():
    st.session_state['all_info']=False

# Sidebar for selecting the LLM and entering the API keys
with st.sidebar:
    llm_name_value = st.selectbox('assistant LLM', 
                        ('llama-3.3-70b-versatile','gemini-2.0-flash', 'gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo','gemma2-9b-it'), 
                        index=None, 
                        placeholder='llama-3.3-70b-versatile')

    if llm_name_value in ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo']:
        openai_api_key = st.text_input("OpenAI API Key ([Get an OpenAI API key](https://platform.openai.com/account/api-keys))", 
                                    key="openai_api_key", 
                                    type="password",
                                    )
    elif llm_name_value in ['llama-3.3-70b-versatile','gemma2-9b-it']:
        groq_api_key = st.text_input("Groq API Key ([Get an Groq API key](https://console.groq.com/keys))", 
                                   key="groq_api_key", 
                                   type="password",
                                   )
    elif llm_name_value in ['gemini-2.0-flash']:
        gemini_api_key = st.text_input ("Gemini API Key ([Get Gemini API Key](https://aistudio.google.com/apikey))",
                                        key="gemini_api_key", 
                                        type="password",
                                        )
    if llm_name_value in ['llama-3.3-70b-versatile','gemma2-9b-it']:
        st.session_state['llm_name'] = llm_name_value
    elif llm_name_value in ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo']:
        st.session_state['llm_name'] = llm_name_value
    elif llm_name_value in ['gemini-2.0-flash']:
        st.session_state['llm_name'] = llm_name_value
    else:
        st.session_state['llm_name'] = 'gpt-4o'

    if not openai_api_key:
        if llm_name_value in ["gpt-4o", "gpt-4o-mini", 'gpt-3.5-turbo']:
            st.info("Please add your OpenAI API key to continue.", icon="🗝️")

    if not groq_api_key:
        if llm_name_value in ['llama-3.3-70b-versatile','gemma2-9b-it']:
            st.info("Please add your Groq API key to continue.", icon="🗝️")
    
    if not gemini_api_key:
        if llm_name_value in ['gemini-2.0-flash']:
            st.info("Please add your Gemini API key to continue.", icon="🗝️")

# Check if all necessary material information is provided
if not (st.session_state['all_info']):
    st.info("Please provide all necessary material information on the Intro page")

# Create LLM client if all necessary information is provided    
if (openai_api_key or groq_api_key or gemini_api_key) and st.session_state['all_info']:
    # Create LLM client.
    if st.session_state['llm_name'] in ["gpt-4o", "gpt-4o-mini", 'gpt-3.5-turbo']:
        client = create_client(llm_name=st.session_state['llm_name'], api_key=st.session_state['openai_api_key']) 
    elif st.session_state['llm_name'] in ['llama-3.3-70b-versatile','gemma2-9b-it']:
        client = create_client(llm_name = st.session_state['llm_name'], api_key=st.session_state['groq_api_key']) 
    elif st.session_state['llm_name'] in ['gemini-2.0-flash']:
        client = create_client(llm_name=st.session_state['llm_name'], api_key=st.session_state['gemini_api_key']) 

    st.markdown('** Ask the agent to generate an input QE SCF file for the compound you uploaded**')
    # Create a session state variable to store the chat messages. This ensures that the
    # messages persist across reruns.
    
    # Create task for the agent
    input_file_schema, task = create_task(st.session_state['structure'],
                                          st.session_state['kspacing'],
                                          st.session_state['list_of_element_files'],
                                          st.session_state['cutoffs'])
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages=[{"role": "system", "content": task}]
        st.session_state.messages.append({"role": "system", "content": input_file_schema})
    else:
        for message in st.session_state.messages:
            if message['role']=='system':
                message['content']=task+' '+input_file_schema

    # Display chat history
    for message in st.session_state.messages:
        if(message["role"]=="user"):
            with st.chat_message("user"):
                st.markdown(message["content"])
        elif(message["role"]=="assistant"):
            with st.chat_message("assistant"):
                st.markdown(message["content"])

    # Get user input
    if prompt := st.chat_input("Do you have any questions?"):

        # Store and display the current prompt.
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate a response using one of the LLMs
        stream = generate_llm_response(llm_name=st.session_state['llm_name'],
                                       messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                                       client = client)
           
        # Stream the response to the chat using `st.write_stream`, then store it in 
        # session state.
        with st.chat_message("assistant"):
            response = st.write_stream(stream)
        
        st.session_state.messages.append({"role": "assistant", "content": response})