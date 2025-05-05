## Instructions

Follow these steps to set up and run the application:

```bash
# 1. Navigate to the project directory
cd path/to/project-directory

# 2. Create and activate a virtual environment
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install Ollama (https://ollama.com/download)
# Follow instructions for your OS, then continue below once installed

# 5. Pull the LLaMA 3.1 model
ollama pull llama3.1

# 6. Start the FastAPI backend server
python -c "import uvicorn; uvicorn.run('main:app', host='0.0.0.0', port=8000, workers=1)"

# 7. In a new terminal, run the Streamlit frontend
streamlit run auto_population.py
```

Streamlit will open a browser. Start interacting with the front-end that will show in your browser.

If you don't have a GPU on your device, the ChatBot will perform much slower.

ctrl+c to shut down back end and front end in terminals.


