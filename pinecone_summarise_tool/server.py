# server.py
from flask import Flask, request, jsonify, send_from_directory
from bs4 import BeautifulSoup
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from load_dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import re, os

load_dotenv()
app = Flask(__name__)

def clean_html_content(html_content: str):
    soup = BeautifulSoup(html_content, 'html.parser')
    for tag in soup(["script", "nav", "footer"]):
        tag.decompose()
    return soup.get_text(separator="\n")

def clean_text(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

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
    pinecone_index = get_pc_index("frogs")

    embedding_model = "multilingual-e5-large"
    embeddings = pc.inference.embed(
        model=embedding_model,
        inputs=[d["text"] for d in data],
        parameters={"input_type": "passage", "truncate": "END"},
    )

    records = []
    for d, e in zip(data, embeddings):
        records.append({
            "id": d["id"],
            "values": e["values"],
            "metadata": {"text": d["text"]}
        })

    pinecone_index.upsert(vectors=records, namespace="web")

@app.route('/summarise', methods=['POST'])
def summarise():
    try:
        url = request.json['url']
        print(f"🛰️ Received URL: {url}")
        loader = WebBaseLoader(url)
        data = loader.load()
        print("✅ Data loaded")

        # Clean & summarise
        cleaned_text = [clean_html_content(page.page_content) for page in data]
        cleaned_scraped_text = clean_text(" ".join(cleaned_text))

        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("Missing GROQ_API_KEY in environment")

        model = ChatGroq(api_key=groq_api_key, model="gemma2-9b-it")

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "you are a useful assistant."),
            ("user", "Summarize the following text extremely informatively, but in the format of a conversation between peter and stewie griffin from family guy. For example, write Peter:... Stewie:..., include unwanted and off-topic interjections from Lois Griffin, Capitalise all of their names. {text}"),
        ])
        formatted_prompt = prompt_template.invoke({"text": cleaned_scraped_text})
        summary = model.invoke(formatted_prompt)
        print("✅ Summary complete")

        return jsonify({'summary': summary.content})
    except Exception as e:
        print("❌ Error:", e)
        return jsonify({'error': str(e)}), 500


@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')


if __name__ == '__main__':
    app.run(debug=False)
