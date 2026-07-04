import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

load_dotenv()

st.set_page_config(
    page_title="Movie Information Extractor",
    page_icon="🎬"
)

model = ChatMistralAI(
    model="mistral-small-latest"
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert Movie Information Extractor.

Analyze the provided movie description and extract only the most useful information.

Requirements:
- Identify the movie title.
- Extract genres.
- Extract director name.
- Extract cast members.
- Extract release year if available.
- Extract IMDb rating or any ratings mentioned.
- Extract music composer if mentioned.
- Identify major themes.
- Identify notable features or strengths.
- Extract important keywords useful for search and recommendations.
- Generate a concise plot summary (2-3 sentences).
- Generate a quick summary in one sentence.
- If any information is not available, write "Not Mentioned".
- Do not add information that is not present in the text.

Output Format:

Movie Name:
Genre:
Director:
Cast:
Release Year:
IMDb Rating:
Music Composer:

Key Themes:
- item 1
- item 2

Notable Features:
- item 1
- item 2

Keywords:
- item 1
- item 2

Plot Summary:

Quick Summary:
"""
    ),
    (
        "human",
        """
Extract information from this paragraph:

{paragraph}
"""
    )
])

st.title("🎬 Movie Information Extractor")

paragraph = st.text_area(
    "Enter Movie Description",
    height=250
)

if st.button("Extract Information"):
    if paragraph.strip():
        try:
            final_prompt = prompt.invoke(
                {"paragraph": paragraph}
            )

            response = model.invoke(final_prompt)

            st.subheader("Extracted Information")
            st.markdown(response.content)

        except Exception as e:
            st.error(f"Error: {str(e)}")

    else:
        st.warning("Please enter a movie description.")