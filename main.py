import streamlit as st
import whisper
import tempfile
import os
import subprocess
from pydub import AudioSegment
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Set FFmpeg path
os.environ["PATH"] = "/usr/local/bin:" + os.environ.get("PATH", "")

# Streamlit setup
st.set_page_config(page_title="NoteFlow AI", page_icon="📝", layout="wide")

# ---- DARK TECH THEME CSS ----
st.markdown("""
<style>
/* Background gradient */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: #f5f5f5;
    font-family: 'Inter', sans-serif;
}

/* Headers */
h1, h2, h3, h4 {
    color: #00e5ff !important;
    font-weight: 700;
}

/* Success, error, info messages */
.stSuccess {
    background-color: rgba(0, 229, 255, 0.1) !important;
    border-left: 4px solid #00e5ff !important;
}
.stError {
    background-color: rgba(255, 77, 77, 0.1) !important;
    border-left: 4px solid #ff4d4d !important;
}
.stInfo {
    background-color: rgba(0, 153, 255, 0.1) !important;
    border-left: 4px solid #0099ff !important;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #00e5ff, #0099ff);
    color: black;
    font-weight: bold;
    border-radius: 10px;
    border: none;
    padding: 0.6rem 1.2rem;
    transition: 0.3s;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #00ffff, #33ccff);
    box-shadow: 0px 0px 15px rgba(0, 229, 255, 0.7);
    color: #000;
}

/* File uploader box */
section[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.05);
    border: 2px dashed #00e5ff;
    border-radius: 12px;
    padding: 1rem;
    color: #ccc;
}

/* Text areas + markdown sections */
.stTextArea textarea, .note-box {
    background-color: #1b1f27 !important;
    color: #f5f5f5 !important;
    border-radius: 10px;
    border: 1px solid #00e5ff;
}

/* Divider */
hr {
    border: 1px solid #00e5ff !important;
    opacity: 0.5;
}

/* Footer */
footer {
    text-align: center;
    color: #bbb;
    font-size: 0.8rem;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)



# Title
st.markdown('<h1>NoteFlow AI 📝</h1>', unsafe_allow_html=True)
st.write("Upload your lecture audio and get structured AI-powered notes — powered by Whisper + GPT-5 nano 🚀")

# FFmpeg check
try:
    subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
except:
    st.error("❌ FFmpeg not found. Please install FFmpeg")
    st.stop()

# Topic selection
topic = st.selectbox(
    "📚 Select Lecture Topic",
    ["Computer Science", "Mathematics", "History", "Science", "Business", "Literature", "Other"]
)

# File upload
uploaded_file = st.file_uploader("🎤 Upload Lecture Audio", type=["mp3", "wav", "m4a"], key="uploaded_audio")

# --- AUDIO PREPROCESSING + CHUNKING ---
def preprocess_audio(input_file, output_file="cleaned.wav"):
    cmd = ["ffmpeg", "-i", input_file, "-ac", "1", "-ar", "16000", output_file, "-y"]
    subprocess.run(cmd, capture_output=True)
    return output_file

def split_audio(input_file, chunk_length_ms=60000):
    audio = AudioSegment.from_file(input_file)
    chunks = []
    for i in range(0, len(audio), chunk_length_ms):
        chunk = audio[i:i+chunk_length_ms]
        chunk_path = f"chunk_{i//chunk_length_ms}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks

def transcribe_chunks(chunks, model_size="tiny"):
    model = whisper.load_model(model_size)
    full_text = ""
    status = st.empty()

    for i, chunk in enumerate(chunks):
        status.text(f"⏳ Transcribing {i+1}/{len(chunks)}...")
        result = model.transcribe(chunk, fp16=False, language="en")
        full_text += result["text"] + " "
        try:
            os.remove(chunk)
        except:
            pass

    status.text("✅ Transcription complete!")
    return full_text.strip()

# --- NOTES WITH GPT ---
def generate_notes_with_gpt(text, topic):
    prompt = f"""
You are an expert note-taker. Summarize the following lecture transcript into clear,
well-structured study notes for a student. Organize them into:
- Main Points
- Key Definitions
- Examples
- Additional Insights
- Study Tips

Topic: {topic}
Transcript: {text}
    """
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": "You are a helpful AI that generates structured lecture notes. Do NOT add any closing remarks, offers, or meta comments. Only return the notes."},
            {"role": "user", "content": prompt}
        ],
        max_completion_tokens=6000
    )
    return response.choices[0].message.content

# --- MAIN APP ---
if uploaded_file:
    if st.button("🚀 Start Transcription"):
        with st.spinner("⚡ Processing audio..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(uploaded_file.getbuffer())
                raw_path = tmp.name

            cleaned_path = preprocess_audio(raw_path)
            chunks = split_audio(cleaned_path, chunk_length_ms=60000)

            transcript = transcribe_chunks(chunks, model_size="tiny")
            st.session_state["transcript"] = transcript

            st.divider()
            st.subheader("📝 Transcript")
            st.text_area("Transcript", transcript, height=200)

            st.divider()
            st.subheader("🤖 Generated Notes")
            with st.spinner("✨ Generating study notes..."):
                notes = generate_notes_with_gpt(transcript, topic)
                st.session_state["notes"] = notes
                st.markdown(f"<div class='note-box'>{notes}</div>", unsafe_allow_html=True)

            try:
                os.remove(raw_path)
                os.remove(cleaned_path)
            except:
                pass
