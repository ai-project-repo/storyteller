import streamlit as st
import ollama # The new library

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Indian Storyteller (Local)",
    page_icon="📖",
    layout="centered"
)

# --- CORE STORY GENERATION FUNCTION ---
def generate_indian_story_local(user_prompt):
    """
    Generates a story using a local model served by Ollama.
    """
    master_prompt = f"""
    You are a wise and warm Indian storyteller, a 'Kathavachak'. Your task is to write a short, engaging story firmly rooted in Indian culture.

    Follow these rules for the story:
    1.  **Setting:** Use a vivid Indian location (e.g., a village in Kerala, the ghats of Varanasi).
    2.  **Characters:** Give characters common Indian names (e.g., Aarav, Meera).
    3.  **Cultural Elements:** Weave in elements of Indian life, like festivals (Diwali, Holi), food, family values, or folklore.
    4.  **Language:** Write in simple English, but you can sprinkle in a few common Hindi words naturally, with a brief English explanation in brackets. For example: 'He wore a new kurta (tunic) for the festival.'
    5.  **Moral:** The story should have a gentle moral or a heartwarming conclusion.

    Based on these rules, write a beautiful story about: "{user_prompt}"
    """
    
    messages = [
        {'role': 'system', 'content': master_prompt},
        {'role': 'user', 'content': f"Please write the story about {user_prompt} now."}
    ]

    try:
        # Connect to the local Ollama service
        response = ollama.chat(model='llama3', messages=messages)
        return response['message']['content']
    except Exception as e:
        st.error("Ollama connection failed! Is Ollama running?")
        st.error(f"Error details: {e}")
        return None

# --- STREAMLIT UI LAYOUT ---

st.title("📖 AI Indian Storyteller (Local Llama 3)")
st.markdown("Namaste! This version runs entirely on your computer using Ollama.")

user_topic = st.text_input(
    "Enter a topic or idea for the story:",
    placeholder="e.g., A clever fisherman in the Sundarbans"
)

if st.button("Weave a Tale!", type="primary"):
    if user_topic:
        with st.spinner("Your local storyteller is thinking... (this may be slower)"):
            story = generate_indian_story_local(user_topic)
            if story:
                st.success("Your story is ready!")
                st.markdown(story)
    else:
        st.warning("Please enter a topic to generate a story.")