from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os


def get_llm():
    """Create the Mistral client used for transcript analysis."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is not set in the environment or .env file")
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=api_key,
        temperature=0.2,
    )


def _split_transcript(transcript: str) -> list[str]:
    if not transcript or not transcript.strip():
        raise ValueError("The transcript is empty; nothing can be summarized.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=8_000, chunk_overlap=500)
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    """Create a concise summary, handling long transcripts in stages."""
    llm = get_llm()
    summarize_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert meeting assistant. Summarize the transcript clearly, covering the main topics, outcomes, and next steps."),
        ("human", "Transcript:\n{text}"),
    ])
    chain = summarize_prompt | llm | StrOutputParser()
    partial_summaries = [chain.invoke({"text": chunk}) for chunk in _split_transcript(transcript)]

    if len(partial_summaries) == 1:
        return partial_summaries[0]

    combine_prompt = ChatPromptTemplate.from_messages([
        ("system", "Combine these partial meeting summaries into one concise, coherent summary. Preserve important decisions and action items."),
        ("human", "Partial summaries:\n{text}"),
    ])
    return (combine_prompt | llm | StrOutputParser()).invoke(
        {"text": "\n\n".join(partial_summaries)}
    )


def generate_title(transcript: str) -> str:
    """Generate a short descriptive title for a meeting transcript."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Generate a specific meeting title in at most 10 words. Return only the title."),
        ("human", "Transcript:\n{text}"),
    ])
    return (prompt | get_llm() | StrOutputParser()).invoke(
        {"text": _split_transcript(transcript)[0]}
    ).strip()
