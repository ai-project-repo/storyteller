import os
import streamlit as st
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from gtts import gTTS
import base64
import tempfile

# --- 1. PAGE AND API CONFIGURATION ---

st.set_page_config(
    page_title="AI Indian Storyteller 🇮🇳",
    page_icon="📖",
    layout="wide"
)

# Load API token
load_dotenv()
hf_token = os.getenv('hugging_key') or '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'

# --- 2. SESSION STATE INITIALIZATION ---

if "story" not in st.session_state:
    st.session_state.story = ""
if "model_choice" not in st.session_state:
    st.session_state.model_choice = "Base Model"
if "client" not in st.session_state:
    st.session_state.client = None

def text_to_speech(text):
    """Generates and returns an audio file path from text."""
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name
    except Exception as e:
        st.error(f"Text-to-speech error: {e}")
        return None

def generate_story(client, system_prompt, user_prompt):
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

# --- 3. STREAMLIT UI LAYOUT ---

st.title("📖 AI Indian Storyteller 🇮🇳")
st.markdown("Craft your own tale from the heart of India, now with more creative control!")

# Sidebar for customization options
with st.sidebar:
    st.header("🎨 Customize Your Tale")
    st.session_state.model_choice = st.radio("Choose Your AI Model:", ["Base Model", "Fine-Tuned Model"], key="model_choice_radio")

    if st.session_state.model_choice == "Base Model":
        MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
        st.info(f"Using the base model: `{MODEL_ID}`")
    else:
        # NOTE: This is a placeholder. You must replace this with your actual fine-tuned model's repo ID.
        MODEL_ID = "YOUR_HUGGINGFACE_USERNAME/your-finetuned-model"
        st.info(f"Using a fine-tuned model: `{MODEL_ID}`")
        st.warning("Please replace `YOUR_HUGGINGFACE_USERNAME/your-finetuned-model` with your own model's ID after fine-tuning.")

    try:
        if not hf_token:
            st.error("HUGGINGFACE_API_TOKEN not found. Please set it as a Streamlit secret.")
            st.stop()
        st.session_state.client = InferenceClient(model=MODEL_ID, token=hf_token)
    except Exception as e:
        st.error(f"API configuration error: {e}")
        st.stop()

    st.divider()
    genre = st.selectbox("Choose a Genre:", ["Folktale", "Fantasy", "Mystery", "Sci-Fi", "Romance", "Horror"])
    tone = st.radio("Select a Tone:", ["Heartwarming", "Funny", "Dark & Gritty", "Poetic", "Suspenseful"])
    length = st.select_slider("Desired Story Length:", ["Short (~300 words)", "Medium (~600 words)", "Long (~1000 words)"])
    custom_chars = st.text_area("Define Characters (optional):", placeholder="e.g., 'A brave warrior named Vikram and a wise sorceress named Aisha.'")

    st.divider()
    st.header("💡 Fine-Tuning Your Own Storyteller")
    with st.expander("Learn How"):
        st.markdown(
            """
            To create your own custom storyteller, you can fine-tune a base model like Llama 3 on a dataset of Indian stories. Here's a conceptual overview of the process:

            1.  **Prepare a Dataset:** Collect a dataset of text in a format that your model can learn from. For example, a JSON file with `{"prompt": "A story about a princess...", "completion": "The princess journeyed through the desert..."}`. The better the data, the better the fine-tuned model.

            2.  **Use Hugging Face Tools:** Hugging Face provides libraries like `transformers`, `peft`, and `trl` that make fine-tuning much easier. You can use their `Trainer` class to handle the training loop. This is where you'd run the code to train your model on your dataset.

            3.  **Train the Model:** The training process involves feeding your dataset to the model and letting it learn from your examples. This step requires a powerful GPU and can take a long time, but it results in a model that specializes in your desired style of storytelling. 

            4.  **Push to the Hub:** Once training is complete, you can push the fine-tuned model to your own repository on the Hugging Face Hub. It will have a unique `repo_id` like `YOUR_HUGGINGFACE_USERNAME/your-finetuned-model`.

            Once you've done this, simply select "Fine-Tuned Model" in the sidebar and replace the placeholder model ID to use your own custom AI!
            """
        )

# Main page for story generation
st.header("✨ Let's Create a Story")
topic = st.text_input("What is the main idea or topic of your story?", placeholder="e.g., A hidden temple in the Himalayas")

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
            new_story = generate_story(st.session_state.client, master_prompt, topic)
            if new_story:
                st.session_state.story = new_story

# --- 4. STORY DISPLAY, CONTINUATION, AND TTS ---

if st.session_state.story:
    st.divider()
    st.subheader("Your Generated Tale")

    # Text-to-Speech
    if st.button("▶️ Listen to the Story"):
        with st.spinner("Generating audio..."):
            audio_file = text_to_speech(st.session_state.story)
            if audio_file:
                with open(audio_file, "rb") as f:
                    audio_bytes = f.read()
                    base64_audio = base64.b64encode(audio_bytes).decode()
                    audio_html = f'<audio autoplay="true" controls="controls"><source src="data:audio/mp3;base64,{base64_audio}" type="audio/mp3"></audio>'
                    st.markdown(audio_html, unsafe_allow_html=True)
                os.remove(audio_file)  # Clean up

    st.markdown(st.session_state.story)

    # Continuation and Editing Section
    st.divider()
    st.subheader("✍️ Continue or Edit the Story")
    modification_prompt = st.text_area("How would you like to change or extend the story?",
                                        placeholder="e.g., 'Now, introduce a villain who steals the magical jewel.' or 'Make the ending happier.'")

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
                updated_story = generate_story(st.session_state.client, continuation_system_prompt, modification_prompt)
                if updated_story:
                    st.session_state.story = updated_story
                    st.rerun()
