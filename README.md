# 🚀 The Feedback Cruncher

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://feedback-cruncher-jwuhpfmpggxudtljlasplw.streamlit.app/)

**Turn messy customer feedback into actionable product insights in seconds.**

The Feedback Cruncher is an AI-powered analytics tool designed for Product Managers. It ingests raw customer reviews (CSV), uses LLMs to analyze sentiment and extract pain points, and visualizes the data via an interactive dashboard.

### 🌐 [Live Demo](https://feedback-cruncher-jwuhpfmpggxudtljlasplw.streamlit.app/)

---

## 🌟 Key Features

* **📂 Drag-and-Drop Upload:** Works with any CSV file containing customer reviews.
* **🧠 AI-Powered Analysis:** Uses OpenAI to categorize sentiment (Positive, Negative, Neutral) and identify specific feedback themes.
* **🔧 Dynamic Column Selection:** Automatically detects CSV headers and lets you choose which column contains the feedback text—no manual formatting required.
* **🎛 Smart Sampling:** Includes a slider to control sample size, preventing API timeouts on large datasets (e.g., analyze 50 random comments from a file of 2,000).
* **📊 Interactive Visualizations:**
    * Sentiment Distribution (Pie Chart)
    * Volume by Date (Bar Chart)
    * Automated "Pain Points" Summary

---

## 🛠 Tech Stack

* **Frontend:** Streamlit
* **Data Processing:** Pandas
* **Visualization:** Plotly
* **AI/LLM:** OpenAI API
* **Environment:** Python-dotenv

---

## 🚀 How to Run Locally

If you want to run this on your own machine:

### 1. Clone the Repository
```bash
git clone [https://github.com/notlevangorga/feedback-cruncher.git](https://github.com/notlevangorga/feedback-cruncher.git)
cd feedback-cruncher