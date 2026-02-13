"""
Product Feedback Analyzer - 'The Feedback Cruncher'
Streamlit app for Product Managers to analyze customer reviews with AI.
"""

import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Page config
st.set_page_config(page_title="The Feedback Cruncher", page_icon="📊", layout="wide")

# Subtle custom styling to make the upload dropzone more visually obvious
st.markdown(
    """
    <style>
    div[data-testid="stFileUploaderDropzone"] {
        border: 2px dashed #4f46e5;
        background-color: #f9fafb;
        transition: border-color 0.15s ease, background-color 0.15s ease, box-shadow 0.15s ease;
    }
    div[data-testid="stFileUploaderDropzone"]:hover {
        border-color: #4338ca;
        background-color: #eef2ff;
        box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title & Intro
st.title("The Feedback Cruncher")
st.markdown(
    "Turn raw customer feedback into clear tags, sentiment, and an executive summary in seconds."
)
st.markdown("### How it works")
st.markdown(
    "- **1. Upload** a CSV of customer feedback\n"
    "- **2. Configure** which column to analyze and how many rows\n"
    "- **3. Review** AI-generated tags, sentiment charts, and a summary of pain points"
)
st.markdown("---")

# File uploader
uploaded_file = st.file_uploader(
    "📂 Upload a feedback CSV",
    type=["csv"],
    help="CSV file with at least one column containing free-text customer feedback (e.g. 'comment', 'review').",
)

if uploaded_file is not None:
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)

    # Step 1: Configure your data
    st.markdown("### 1. Configure your data 📁")
    st.caption(
        f"Loaded **{len(df):,}** rows and **{len(df.columns):,}** columns from `{uploaded_file.name}`."
    )

    # Let the user choose which column contains the feedback text
    st.subheader("Select feedback column")
    feedback_column = st.selectbox(
        "Which column contains the feedback text?",
        options=df.columns,
        help="Choose the column that contains the free-text customer feedback you want to analyze.",
    )

    # Data preview for the selected column
    st.subheader("Data preview")
    st.dataframe(
        df[[feedback_column]].head(5).rename(columns={feedback_column: "Feedback"}),
        use_container_width=True,
    )
    st.caption("Showing the first 5 rows of the selected feedback column.")
    st.markdown("---")

    # Step 2: Choose sample size
    st.markdown("### 2. Choose sample size 🎯")
    max_rows = min(50, len(df))
    min_rows = min(5, max_rows)
    default_sample = max_rows
    sample_size = st.slider(
        "Sample size (number of feedback rows to analyze)",
        min_value=min_rows,
        max_value=max_rows,
        value=default_sample,
        step=1,
        help="Up to 50 rows for quick, low-cost analysis. Larger samples give more robust insights but can take slightly longer.",
    )
    st.caption(
        f"The app will analyze the first **{sample_size}** rows from the selected feedback column."
    )
    st.markdown("---")

    # Step 3: Run AI analysis
    st.markdown("### 3. Run AI analysis 🤖")

    # Analyze Feedback button
    if st.button("Analyze feedback", type="primary"):
        comments = df[feedback_column].astype(str).head(sample_size).tolist()

        if not comments:
            st.warning("No feedback rows found to analyze. Please check the selected column and sample size.")
            st.stop()

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error(
                "OPENAI_API_KEY not set. Add it to a .env file in the same folder as app.py."
            )
            st.stop()

        with st.spinner(f"Analyzing {len(comments)} feedback rows with AI..."):
            client = OpenAI(api_key=api_key)

            BATCH_SIZE = 25
            results = []
            for start in range(0, len(comments), BATCH_SIZE):
                batch = comments[start : start + BATCH_SIZE]
                comments_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(batch)])
                prompt = f"""For each of the customer feedback comments below, you must assign exactly one "tag" and one "sentiment".

Tags (use exactly one per comment): Bug, Feature Request, UX/UI, Pricing, Praise
Sentiment (use exactly one per comment): Positive, Neutral, Negative

Comments:
{comments_text}

Return ONLY a valid JSON object with this exact structure:
{{
  "results": [
    {{"tag": "Bug", "sentiment": "Negative"}},
    ...
  ]
}}"""

                raw = ""
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a product feedback analyst. "
                                    "You MUST respond with ONLY a single valid JSON object matching the requested schema. "
                                    "Do not include any extra text, explanations, or markdown."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.2,
                        response_format={"type": "json_object"},
                        timeout=30,
                    )
                    raw = response.choices[0].message.content.strip()
                    raw = raw.strip()
                    parsed = json.loads(raw)
                    batch_results = parsed.get("results", [])
                    if not isinstance(batch_results, list):
                        raise json.JSONDecodeError(
                            '"results" field is not a list', raw, 0
                        )
                    results.extend(batch_results)
                except json.JSONDecodeError as e:
                    st.error(f"AI returned invalid JSON: {e}. Raw: {raw[:500] if raw else 'empty'}...")
                    st.stop()
                except Exception as e:
                    st.error(f"OpenAI error: {e}")
                    st.stop()

        # Ensure we have one result per comment (AI might return fewer)
        tags = []
        sentiments = []
        valid_tags = {"Bug", "Feature Request", "UX/UI", "Pricing", "Praise"}
        valid_sentiments = {"Positive", "Neutral", "Negative"}
        for i, item in enumerate(results):
            if i >= len(comments):
                break
            tag = item.get("tag", "Praise")
            sentiment = item.get("sentiment", "Neutral")
            if tag not in valid_tags:
                tag = "Praise"
            if sentiment not in valid_sentiments:
                sentiment = "Neutral"
            tags.append(tag)
            sentiments.append(sentiment)
        while len(tags) < len(comments):
            tags.append("Praise")
            sentiments.append("Neutral")

        df_analysis = df.iloc[: len(comments)].copy()
        df_analysis["Tag"] = tags[: len(df_analysis)]
        df_analysis["Sentiment"] = sentiments[: len(df_analysis)]

        # Prepare aggregations once
        tag_counts = pd.Series(df_analysis["Tag"]).value_counts().reset_index()
        tag_counts.columns = ["Tag", "Count"]
        sentiment_counts = pd.Series(df_analysis["Sentiment"]).value_counts().reset_index()
        sentiment_counts.columns = ["Sentiment", "Count"]

        st.success("Analysis complete.")

        overview_tab, charts_tab, table_tab = st.tabs(
            ["Overview ⭐", "Charts 📊", "Full table 📋"]
        )

        with overview_tab:
            st.subheader("Key metrics")
            st.caption("High-level snapshot of what your customers are saying in this sample.")
            total_rows = len(df_analysis)
            most_common_tag = tag_counts.iloc[0]["Tag"] if not tag_counts.empty else "–"
            negative_count = (
                int(
                    sentiment_counts.loc[
                        sentiment_counts["Sentiment"] == "Negative", "Count"
                    ].sum()
                )
                if not sentiment_counts.empty
                else 0
            )
            negative_pct = (negative_count / total_rows * 100) if total_rows else 0

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Feedback analyzed", f"{total_rows:,}")
            with col2:
                st.metric("Most common tag", most_common_tag)
            with col3:
                st.metric("Negative feedback", f"{negative_pct:.0f}%")

            st.markdown("---")
            st.subheader("Summary: Biggest pain points")
            summary_prompt = f"""Based on these customer comments and their tags/sentiments, write exactly 3 bullet points summarizing the biggest pain points. Be concise and actionable.

Comments (with tag/sentiment):
{chr(10).join([f"- {c} [Tag: {t}, Sentiment: {s}]" for c, t, s in zip(comments[:50], tags[:50], sentiments[:50])])}

Return only the 3 bullet points, one per line, each starting with "- "."""

            try:
                with st.spinner("Generating summary..."):
                    summary_response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": summary_prompt}],
                        temperature=0.3,
                        timeout=30,
                    )
                summary_text = summary_response.choices[0].message.content.strip()
                st.markdown(summary_text)
            except Exception as e:
                st.warning(f"Could not generate summary: {e}")

        with charts_tab:
            st.subheader("Tag distribution")
            st.caption("How often each tag appears across the analyzed feedback.")
            fig_tags = px.bar(
                tag_counts,
                x="Tag",
                y="Count",
                color="Count",
                color_continuous_scale="Blues",
            )
            fig_tags.update_layout(xaxis_title="Tag", yaxis_title="Count", showlegend=False)
            st.plotly_chart(fig_tags, use_container_width=True)

            st.subheader("Sentiment distribution")
            st.caption("Overall sentiment mix for the analyzed feedback.")
            fig_sentiment = px.pie(
                sentiment_counts,
                values="Count",
                names="Sentiment",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            st.plotly_chart(fig_sentiment, use_container_width=True)

        with table_tab:
            st.subheader("Full analysis table")
            table_df = df_analysis[[feedback_column, "Tag", "Sentiment"]].rename(
                columns={feedback_column: "Feedback"}
            )
            csv_data = table_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download analysis as CSV",
                data=csv_data,
                file_name="feedback_analysis.csv",
                mime="text/csv",
            )
            st.dataframe(
                table_df,
                use_container_width=True,
            )

else:
    st.info(
        "Start by uploading a CSV file that contains at least one column of free-text customer feedback. "
        "You will then be able to choose which column to analyze and how many rows to sample."
    )
