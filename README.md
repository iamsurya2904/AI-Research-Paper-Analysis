🎓 EduSmart AI: Research Paper Analyzer & Career Guidance Platform

EduSmart AI is a modern, Streamlit-powered intelligent assistant built for students, educators, and researchers. It features a dual-purpose system:
1. AI-Enhanced Research Paper Analyzer** for intelligent summarization and chunk-wise explanation.
2. AI Career Guidance Engine** to recommend tech roles based on your skills profile.

🚀 Key Features

📄 Research Paper Analyzer
- 🧠 Uses LLMs (Gemini, GPT-4, Claude, LLaMA, Mixtral) for advanced summarization.
- 📊 Chunk-wise explanation with math, examples, and prerequisite highlights.
- 🛠️ Async/Sync modes for flexible execution.
- 🧾 Summarization strategies: `map_reduce`, `refine`, `stuff`.
- 🔍 Customizable explanation difficulty from High School to Expert.
- 🌐 Gemini-powered final analysis and recommendations.

💼 Career Guidance System
- 🎯 Predicts ideal IT career roles based on your skill levels (0–100).
- 🔢 Uses a trained ML model (`careerlast.pkl`) and dataset (`dataset9000.csv`).
- 📊 Displays top 5 matching careers with confidence scores.
- 📈 Visual comparison of your skills vs industry averages.
- 📚 Course suggestions with direct navigation to related resources.

🧠 Supported LLMs (via APIs)

| Model            | Provider   |
|------------------|------------|
| Gemini 1.5 Pro   | Google     |
| GPT-4 Turbo      | OpenAI     |
| Claude 3 Opus    | Anthropic  |
| LLaMA 3 (70B)    | Groq       |
| Mixtral-8x7B     | Groq       |
| Gemma 2 (9B)     | Groq       |

> **Note**: Multiple providers are integrated dynamically for optimal performance and model selection.

📁 Project Structure

EduSmartAI/
│
├── app.py # Streamlit UI with both main modules
├── LLMSelect.py # Model loader utility
├── asyncExplainer.py # Async LLM explanation generator
├── preprocessor.py # PDF text extractor and preprocessor
├── careerlast.pkl # Pickled ML model for career prediction
├── dataset9000.csv # Dataset used for training skill-to-career mapping
├── requirements.txt # Python dependencies
└── README.md # Project overview

yaml
Copy
Edit
🔧 Getting Started
1. Clone the Repository

```bash
git clone https://github.com/yourusername/edusmart-ai.git
cd edusmart-ai
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Setup API Keys
Set the following environment variables (either in .env or your terminal):

bash
Copy
Edit
export GROQ_API_KEY=your_groq_key
export OPENAI_API_KEY=your_openai_key
export GOOGLE_API_KEY=your_google_key
Or directly modify app.py as needed.

4. Run the App
bash
Copy
Edit
streamlit run app.py
🧪 Example Use Cases
📚 Analyze a dense research paper with ease and get explanations at your understanding level.

🧑‍🎓 Guide yourself (or students) toward suitable tech careers using a data-driven ML model.

🔢 Understand prerequisites and math behind AI/ML, Cybersecurity, or DS topics.

🌐 API Integrations
Google Generative AI (gemini)

OpenAI GPT (3.5, 4, 4o, Turbo)

Anthropic Claude

Groq LLaMA/Mixtral

Mistral & Gemma

🛠️ Dependencies
text
Copy
Edit
streamlit
google-generativeai
aiohttp
pandas
numpy
scikit-learn
plotly
PyMuPDF (if used in preprocessor)
Install using pip install -r requirements.txt

📌 Future Scope
🔄 Add document chunk cross-referencing.

🎯 Personalized upskilling recommendations.

🌐 Connect with job portals or resume analysis tools.

🤖 Add voice-based interaction and feedback loops.

👤 Author
Surya.R
Cybersecurity & CSE Enthusiast | 3rd Year B.Tech
LinkedIn | TryHackMe

📜 License
This project is licensed under the MIT License. See LICENSE for details.

“Empowering education with explainable AI — one paper and one career at a time.” 🌱
