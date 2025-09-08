import os
from typing import List
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import json
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.chat_models import ChatOllama
HISTORY_FILE = "chat_history.json"
PDF_PATH = os.environ.get("RAG_PDF_PATH", "solarcell-basic-knowledge-SolarHub.pdf")
CHROMA_DIR = os.environ.get("CHROMA_DIR", "./chroma_db")
EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")
RETRIEVAL_K = int(os.environ.get("RETRIEVAL_K", "3"))

CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET", "31727230e135e3a2c2939ea912dd7c0d")
CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN", "T4Y2PCaIIMqn6Xp258iY0yhsnPpKcyuvvmuy5N8AOZY0zMDIpJeCclUh253euPEXoPwX22+cyR8KZudu85057GhxD7DQ4zDLAtrCXC0BhKmuCerVGZhBTE6A0VeYT6Odi+nGEFltTA6Ou+gb4z0P3QdB04t89/1O/w1cDnyilFU=")
#---------------------------------------------------------------#

def build_chat_llm():
    model_name = os.environ.get("OLLAMA_MODEL", "gemma3:latest")
    chat_llm = ChatOllama(model=model_name)
    print(f"[LLM] Using Ollama model: {model_name}")
    return chat_llm

def build_prompt(history: List[dict],context: str, question: str) -> str:
    history_lines = "\n".join(f"User: {turn['user']}\nBot: {turn['bot']}" for turn in history)
    return f"""
    Previous Conversation:
{history_lines}

Context:
{context}

Role: You are an engineer.
Task:
- Use a warm and friendly tone
- Answer in Thai language
- Summarize the information clearly and concisely
- Make it easy to understand, even for beginners
- Include relevant emojis such as 🔋☀️🔌 when appropriate

Question: {question}
Answer:
""".strip()
def load_chat_history(user_id: str, max_turns: int = 5) -> List[dict]:
    if not os.path.exists(HISTORY_FILE):
        return []

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)

    return history.get(user_id, [])[-max_turns:]  # ดึงบทสนทนาแค่ล่าสุด n รอบ

def make_rag_answer(vectorstore: Chroma, chat_llm: ChatOllama,user_id:str, question: str, k: int = 5) -> str:
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs: List[Document] = retriever.get_relevant_documents(question)
    context = "\n\n---\n\n".join(d.page_content for d in docs) if docs else "[No document found]"
    history = load_chat_history(user_id=user_id)
    prompt = build_prompt(history=history,context=context, question=question)
    response = chat_llm.invoke(prompt)
    answer = getattr(response, "content", None) or str(response)
    return answer.strip() if answer else "[ERROR] Empty response from LLM."
def save_chat_history(user_id: str, message: str, response: str):
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = {}

    history.setdefault(user_id, []).append({"user": message, "bot": response})

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
app = Flask(__name__)
line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(CHANNEL_SECRET)

@app.route("/", methods=["POST"]) 
def callback():
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return "OK"

@handler.add(MessageEvent, message=TextMessage) 
def handle_message(event: MessageEvent):
    user_id = event.source.user_id
    user_text = (event.message.text or "").strip()
    if not user_text:
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="(empty message)"))
        return

    if user_text.lower() in {"/help", "help"}:
        help_msg = (
            "Hi! Send me a question about the PDF and I'll answer using RAG.\n\n"
            "Commands:\n"
            "- /source : show PDF + embedding info\n"
            "- /id : echo message id\n"
        )
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=help_msg))
        return

    if user_text.lower() == "/source":
        info = f"Indexed PDF: {os.path.basename(PDF_PATH)}\nEmbeddings: {EMBED_MODEL_NAME}\nTop-k: {RETRIEVAL_K}"
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=info))
        return

    if user_text.lower() == "/id":
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=f"msg id: {event.message.id}"))
        return

    answer = make_rag_answer(app.config["VECTORSTORE"], app.config["CHAT_LLM"], user_id, question=user_text,k=RETRIEVAL_K)
    if len(answer) > 1900:
        answer = answer[:1900] + "\n… (truncated)"
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=answer))
    save_chat_history(user_id, user_text, answer)

if __name__ == "__main__":
    print("[BOOT] Loading vectorstore…")
    embedding = SentenceTransformerEmbeddings(model_name=EMBED_MODEL_NAME)
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)

    print("[BOOT] Initializing chat LLM…")
    chat_llm = build_chat_llm()

    app.config["VECTORSTORE"] = vectorstore
    app.config["CHAT_LLM"] = chat_llm

    port = int(os.environ.get("PORT", "5000"))
    print(f"[RUN] Flask listening on 0.0.0.0:{port}")
    app.run(port=port)
