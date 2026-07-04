from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

class Movie(BaseModel):
    title: str
    release_year: Optional(int)
    grnre: List[str]
    director: Optional[str]
    cast: List[str]
    rating: Optional(float)
    summary: str

parser = PydanticOutputParser(pydantic_object=Movie)
load_dotenv()

model = ChatMistralAI(model="mistral-small-2506")

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

para = input("Give your paragraph: ")

final_prompt = prompt.invoke({
    "paragraph": para
})

response = model.invoke(final_prompt)

print(response.content)