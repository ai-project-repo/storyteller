import os
import streamlit as st
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from gtts import gTTS
import io

# --- 1. PAGE AND API CONFIGURATION ---

st.set_page_config(
    page_title="AI Indian Storyteller 🇮🇳",
    page_icon="📖",
    layout="wide"
)

# Load API token and initialize the Inference Client
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
try:
    load_dotenv()
    # It's better to get the token from an environment variable than hardcoding it
    hf_token = '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'
    if not hf_token:
        st.error("HUGGINGFACE_API_TOKEN not found. Please set it in your .env file.")
        st.stop()
    client = InferenceClient(model=MODEL_ID, token=hf_token)
except Exception as e:
    st.error(f"API configuration error: {e}")
    st.stop()

# --- 2. SESSION STATE INITIALIZATION ---

if "story" not in st.session_state:
    st.session_state.story = ""



def generate_story(system_prompt, user_prompt):
    """Generic function to call the LLM."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    try:
        response_stream = client.chat_completion(messages=messages, max_tokens=1500, stream=False)
        return response_stream.choices[0].message.content
    except Exception as e:
        st.error(f"An error occurred while generating the story: {e}")
        return None

# --- 4. STREAMLIT UI LAYOUT ---

st.title("📖 AI Indian Storyteller 🇮🇳")
st.markdown("Craft your own tale from the heart of India, now with more creative control!")

# Sidebar for customization options
with st.sidebar:
    st.header("🎨 Customize Your Tale")
    genre = st.selectbox("Choose a Genre:", ["Folktale", "Fantasy", "Mystery", "Sci-Fi", "Romance", "Horror"])
    tone = st.radio("Select a Tone:", ["Heartwarming", "Funny", "Dark & Gritty", "Poetic", "Suspenseful"])
    length = st.select_slider("Desired Story Length:", ["Short (~300 words)", "Medium (~600 words)", "Long (~1000 words)"])
    custom_chars = st.text_area("Define Characters (optional):", placeholder="e.g., 'A brave warrior named Vikram and a wise sorceress named Aisha.'")

# Main page for story generation
st.header("✨ Let's Create a Story")
topic = st.text_input("What is the main idea or topic of your story?", value="A hidden temple in the Himalayas", placeholder="e.g., A hidden temple in the Himalayas")

if st.button("Weave a New Tale!", type="primary"):
    if not topic:
        st.warning("Please provide a topic for your story.")
    else:
        master_prompt = f"""
        You are a master Indian storyteller. Create a captivating story based on the user's request.
        
        **STORY REQUIREMENTS:**
        - **Genre:** {genre}
        - **Tone:** {tone}
        - **Length:** {length}
        - **Setting:** A vivid Indian location.
        - **Characters:** Give characters Indian names. If the user provided character descriptions, use them: '{custom_chars}'. Otherwise, create your own.
        - **Cultural Elements:** Weave in elements of Indian life, festivals, food, or folklore.
        """
        with st.spinner("The storyteller is gathering inspiration..."):
            new_story = generate_story(master_prompt, topic)
            if new_story:
                st.session_state.story = new_story
                

# --- 5. STORY DISPLAY, CONTINUATION, AND TTS ---

if st.session_state.story:
    st.divider()
    st.subheader("Your Generated Tale")
    st.markdown(st.session_state.story)

    
    # Continuation and Editing Section
    st.divider()
    st.subheader("✍️ Continue or Edit the Story")
    modification_prompt = st.text_area("How would you like to change or extend the story?",
                                       value="Now, introduce a villain who steals the magical jewel.", placeholder="e.g., 'Now, introduce a villain who steals the magical jewel.' or 'Make the ending happier.'")

    if st.button("Update the Tale"):
        if not modification_prompt:
            st.warning("Please provide instructions to update the story.")
        else:
            continuation_system_prompt = f"""
            You are a master story editor. The user has provided a story and an instruction to modify or continue it.
            Your task is to seamlessly rewrite or extend the story based on the user's request, maintaining the original style and context.
            
            **Original Story:**
            {st.session_state.story}
            """
            with st.spinner("The storyteller is reimagining the tale..."):
                updated_story = generate_story(continuation_system_prompt, modification_prompt)
                if updated_story:
                    st.session_state.story = updated_story
                    st.rerun()