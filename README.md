# TTS Dataset Preparation Notebook

This repository contains a Jupyter Notebook for processing audio files, performing speaker diarization, extracting segments, and transcribing them using Whisper and Gemini models. It supports both GPU and CPU environments.

---

## Requirements

- Python **3.10.11**
- GPU with CUDA (optional, for PyTorch acceleration)
- Conda (recommended)

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <repo-folder>
````

---

### 2. Create Conda environment with Python 3.10.11

```bash
conda create -n tts_env python=3.10.11
conda activate tts_env
```

---

### 3. Install PyTorch (choose ONE option)

#### If you have GPU (CUDA 12.1)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### If you are using CPU only

```bash
pip install torch==2.5.1+cpu torchvision==0.20.1+cpu torchaudio==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
```

---

### 4. Install core scientific libraries (pinned versions)

```bash
pip install numpy==1.26.4 scipy==1.15.3 pandas==2.3.3
```

---

### 5. Install Whisper (after PyTorch, before Pyannote)

```bash
pip install openai-whisper
```

---

### 6. Install remaining dependencies (without dependency resolution)

```bash
pip install -r requirements.txt --no-deps
```

---

### 7. Install Pyannote packages (must be installed LAST)

```bash
pip install pyannote-core==6.0.1 --no-deps
pip install pyannote-metrics==4.0.0 --no-deps
pip install pyannote-audio==3.1.1 --no-deps
```

---

### 8. Verify installation

```bash
python -c "import numpy, whisper, pyannote.audio; print('NumPy:', numpy.__version__); print('Whisper OK'); print('Pyannote OK')"
```

If no errors appear, the environment is ready.

---

### 9. Set up environment variables

Create a `.env` file in the repository root.

Add your API keys:

```env
HF_TOKEN=<Your_HuggingFace_Token>
GEMINI_API_KEY=<Your_Gemini_API_Key>
```

---

### 10. Run the Notebook

Open `Notebook.ipynb` in Jupyter.

Execute cells step by step to:

* Perform speaker diarization
* Extract audio segments
* Generate speaker embeddings
* Transcribe audio using Whisper and Gemini

---

## Notes

* GPU availability is detected automatically.
* Speaker embeddings are saved locally for reuse (`speaker_embeddings.pt`).
* Audio segments shorter than `MIN_LEN` are padded or merged to avoid very short clips.

```
```
