import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

# 1. Pydantic Model definition
class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional[float]
    summary: str

parser = PydanticOutputParser(pydantic_object=Movie)

# 2. Page Configuration & Title
st.set_page_config(page_title="AI Movie Info Extractor", page_icon="🎬", layout="centered")
st.title("🎬 AI Movie Information Extractor")
st.caption("Paste a paragraph or review about a movie, and the AI will extract structured data.")

# Load environment variables
load_dotenv()

# Initialize the Mistral model
# (Wrapped in a function with @st.cache_resource to prevent reloading on every run)
@st.cache_resource
def get_model():
    return ChatMistralAI(model="mistral-small-2506")

try:
    model = get_model()
except Exception as e:
    st.error("Could not initialize Mistral AI model. Ensure your MISTRAL_API_KEY is configured in your .env file.")

# 3. Streamlit UI Layout
para = st.text_area(
    "Enter Movie Paragraph/Review:", 
    placeholder="e.g., Inception is a 2010 sci-fi action film written and directed by Christopher Nolan. It stars Leonardo DiCaprio as a professional thief...",
    height=150
)

# Trigger extraction on button click
if st.button("Extract Movie Details", type="primary"):
    if not para.strip():
        st.warning("Please enter some text first!")
    else:
        # Prompt creation
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "Extract Movie information from the paragraph.\n\n{format_instructions}"
            ),
            (
                "human",
                "{paragraph}"
            )
        ])

        final_prompt = prompt.invoke({
            "paragraph": para,
            "format_instructions": parser.get_format_instructions()
        })

        # Run model with a loading spinner animation
        with st.spinner("Extracting data... Please wait."):
            try:
                response = model.invoke(final_prompt)
                
                # Parse the output into the Pydantic format
                parsed_movie = parser.parse(response.content)

                # 4. Display Results cleanly
                st.success("Extraction Complete!")
                
                st.subheader(f"🍿 {parsed_movie.title}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**📅 Release Year:** {parsed_movie.release_year or 'N/A'}")
                    st.markdown(f"**🎬 Director:** {parsed_movie.director or 'N/A'}")
                    st.markdown(f"**⭐ Rating:** {parsed_movie.rating or 'N/A'}")
                
                with col2:
                    st.markdown(f"**🎭 Genres:** {', '.join(parsed_movie.genre)}")
                    st.markdown(f"**👥 Cast:** {', '.join(parsed_movie.cast)}")
                
                st.markdown(f"**📝 Summary:**  \n{parsed_movie.summary}")
                
                # Expandable JSON block for raw output
                with st.expander("View Raw JSON Output"):
                    st.json(parsed_movie.model_dump())

            except Exception as e:
                st.error(f"An error occurred during extraction: {e}")
