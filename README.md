## RAG Recruiter Bot

An AI-powered assistant that helps recruiters and hiring managers search, analyze, and reason over candidate and job data using Retrieval-Augmented Generation (RAG). The app is built with Streamlit and integrates OpenAI models to provide fast, context-aware answers and insights on your recruiting pipeline.

### Key Features

- **RAG-based question answering**: Ask natural-language questions about your candidates, roles, or hiring funnel and get grounded answers based on your own data.
- **Search and filtering**: Retrieve relevant candidates or roles based on skills, experience, location, or any other fields in your dataset.
- **Profile summarization**: Generate concise summaries of CVs or candidate profiles to speed up screening.
- **Comparison and shortlisting**: Compare multiple candidates against a job description to assist with shortlisting decisions.
- **Interactive UI**: Simple Streamlit interface for uploading data, configuring the model, and exploring results.

### Tech Stack

- **Frontend / App framework**: `streamlit`
- **Data handling**: `pandas`
- **Visualization**: `plotly`
- **LLM provider**: `openai`
- **Configuration management**: `python-dotenv`
- **Templating**: `jinja2`

---

## Getting Started

### 1. Prerequisites

- **Python**: 3.9 or later (3.10+ recommended)
- **pip**: latest version
- **OpenAI API key**: You need an active OpenAI account and API key.

### 2. Clone the repository

```bash
git clone https://github.com/<your-username>/rag-recruiter-bot.git
cd rag-recruiter-bot
```

Update the repository URL above to your actual remote if needed.

### 3. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 4. Install dependencies

The main Python dependencies are listed in `requirements.txt`.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Configuration

This project uses environment variables loaded from a `.env` file (listed in `.gitignore` so it is not committed to version control).

Create a file named `.env` at the project root with at least:

```bash
OPENAI_API_KEY=sk-...
```

You can also define optional variables depending on how you configure the app, for example:

- `OPENAI_MODEL` – default OpenAI model to use (e.g., `gpt-4.1-mini`).
- `APP_ENV` – environment name (`local`, `staging`, `prod`, etc.).

Adjust the names and values above to match the variables actually used in your code.

---

## Running the App

From the project root, run the Streamlit app. For example, if your main file is `app.py`:

```bash
streamlit run app.py
```

If your entrypoint file has a different name, update the command accordingly (e.g., `streamlit run main.py`).

Once the app starts, Streamlit will print a local URL (typically `http://localhost:8501`) that you can open in your browser.

---

## Typical Workflow / Usage

1. **Prepare your data**
   - **Candidates**: e.g., a CSV or other structured file with columns such as `name`, `title`, `skills`, `experience`, `location`, etc.
   - **Jobs / roles**: e.g., job descriptions, titles, required skills, and other metadata.
2. **Upload or connect data**
   - Use the Streamlit UI to upload CSV files or connect to your existing data source (depending on how you implement it).
3. **Configure model options**
   - Choose which OpenAI model to use (if exposed in the UI).
   - Optionally tune temperature or other generation parameters.
4. **Ask questions / run analyses**
   - Example questions:
     - “Show me the top 5 candidates for the Data Scientist role in Paris with at least 3 years of experience in Python.”
     - “Summarize this candidate’s profile and highlight any red flags.”
     - “Compare these three candidates against the Senior Backend Engineer job description.”
5. **Review results and iterate**
   - Use the visualizations (via `plotly`) and tables (`pandas` in Streamlit) to explore results and refine your prompts.

---

## RAG Approach (High-Level)

Although implementation details may vary, a typical RAG flow in this project looks like:

1. **Ingestion**: Load structured recruiting data (candidates, roles, notes) into `pandas` DataFrames.
2. **Indexing / retrieval**: Build a representation of documents or rows suitable for similarity search (e.g., embeddings via OpenAI).
3. **Retrieval**: For each user query, retrieve the most relevant items (candidates, roles, notes) from the index.
4. **Generation**: Pass the retrieved context plus the user’s question to an OpenAI model to generate grounded, context-aware answers or summaries.
5. **Display**: Show the answer and supporting evidence in the Streamlit UI, with optional charts or tables.

Adapt this description as needed to match your actual implementation choices (e.g., vector database, in-memory embeddings, etc.).

---

## Project Structure (Example)

The exact structure of your project may differ, but a common layout for this app could look like:

```text
.
├─ app.py                # Main Streamlit app entrypoint
├─ requirements.txt      # Python dependencies
├─ .env                  # Local environment variables (not committed)
├─ data/                 # Example candidate / job data (if provided)
├─ src/
│  ├─ rag_pipeline.py    # Retrieval and generation logic
│  ├─ data_loader.py     # Helpers for loading and cleaning data
│  └─ prompts/           # Jinja2 templates and prompt definitions
└─ README.md             # Project documentation
```

If your structure is different, update this section to reflect the actual layout.

---

## Development Notes

- **Environment isolation**: Use a virtual environment (`.venv`) to avoid dependency conflicts with other projects.
- **Secrets management**: Never commit your `.env` file or API keys. They are already excluded by `.gitignore`.
- **Model costs**: Interacting with OpenAI models incurs cost. Monitor usage in your OpenAI dashboard and consider using smaller/cheaper models during development.

---

## Troubleshooting

- **Streamlit app does not start**
  - Check that your virtual environment is active.
  - Ensure `streamlit` is installed (`pip show streamlit`).
- **Authentication errors from OpenAI**
  - Verify `OPENAI_API_KEY` is correctly set in `.env`.
  - Ensure you have sufficient OpenAI credits and correct API base URL (if using a custom endpoint).
- **Encoding / CSV issues**
  - Confirm your CSV files use UTF-8 encoding.
  - Check that column names in your data match what the code expects.

---

## Contributing

Contributions, ideas, and bug reports are welcome. For now, a simple workflow is:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/my-change`).
3. Make your changes and add tests or example data where helpful.
4. Submit a pull request with a clear description of the change and motivation.

---

## License

Add your preferred license here (for example, MIT, Apache 2.0, or a proprietary license). If this is a public project, you may also want to add a dedicated `LICENSE` file at the repository root.

