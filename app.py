import streamlit as st
import requests

# Set the OpenRouter API endpoint and your API key
API_URL = 'https://api.openrouter.ai/chat'
API_KEY = 'YOUR_API_KEY'  # Replace with your actual API key

st.title('Chatbot Application')

# Function to send a message to the OpenRouter API
def send_message(message):
    response = requests.post(API_URL, json={'input': message}, headers={'Authorization': f'Bearer {API_KEY}'})
    return response.json().get('output', 'Sorry, I could not get a response.')

# Chat message input
if 'messages' not in st.session_state:
    st.session_state.messages = []

message_input = st.text_input('You:', '')

if st.button('Send') and message_input:
    st.session_state.messages.append(('User', message_input))
    response = send_message(message_input)
    st.session_state.messages.append(('Bot', response))
    st.text_input('You:', '', key='new_input')

# Display chat messages
for role, message in st.session_state.messages:
    st.markdown(f'**{role}:** {message}')

# File upload functionality
uploaded_file = st.file_uploader('Upload a file', type=['txt', 'pdf'])
if uploaded_file is not None:
    st.write('Uploaded file:', uploaded_file.name)
    # Add functionality to process the uploaded file here
