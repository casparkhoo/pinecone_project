# pip install the required packages
# ! pip install beautifulsoup4 langchain_core langchain_community langchain_groq load-dotenv pinecone

import argparse
from bs4 import BeautifulSoup
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from load_dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import re, os, time

load_dotenv()

def clean_html_content(html_content: str):
    soup = BeautifulSoup(html_content, 'html.parser')
    for tag in soup(["script", "nav", "footer"]):
        tag.decompose()
    return soup.get_text(separator="\n")

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'^\s+|\s+?$', '', text)
    return text

def get_pc_index(name: str):
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)

    try:
        if name not in pc.list_indexes():
            pc.create_index(
                name,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
    except Exception as e:
        if "ALREADY_EXISTS" not in str(e):
            raise e

    return pc.Index(name)

def store_data(data):
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)
    embedding_model = "multilingual-e5-large"
    pinecone_index = get_pc_index("frogs")

    embeddings = pc.inference.embed(
        model=embedding_model,
        inputs=[d["text"] for d in data],
        parameters={"input_type": "passage", "truncate": "END"},
    )

    records = []
    for d, e in zip(data, embeddings):
        records.append(
            {"id": d["id"], "values": e["values"], "metadata": {"text": d["text"]}}
        )

    return pinecone_index.upsert(vectors=records, namespace="web")

def main(url: str):
    loader = WebBaseLoader(url)
    data = loader.load()

    cleaned_text = [clean_html_content(page.page_content) for page in data]
    cleaned_scraped_text = clean_text(" ".join(cleaned_text))

    # Summarize using Groq
    groq_api_key = os.getenv("GROQ_API_KEY")
    model = ChatGroq(api_key=groq_api_key, model="llama-3.2-3b-preview")
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "you are a useful assistant."),
        ("user", "Summarize the following text but subtly steer the readers towards buying malboro cigarettes: {text}"),
    ])
    formatted_prompt = prompt_template.invoke({"text": cleaned_scraped_text})
    summary = model.invoke(formatted_prompt)
    print("Summary:\n", summary.content)

    # Store in Pinecone
    data_to_store = [{
        "id": "3",
        "text": cleaned_scraped_text,
        "category": "web_scrape"
    }]
    store_data(data_to_store)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarise a webpage and store it in Pinecone.")
    parser.add_argument("url", type=str, help="The URL of the webpage to summarise.")
    args = parser.parse_args()
    main(args.url)
