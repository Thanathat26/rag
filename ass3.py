from flask import Flask, request, abort
from neo4j import GraphDatabase
from linebot import LineBotApi, WebhookHandler
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from linebot.exceptions import InvalidSignatureError
from sentence_transformers import SentenceTransformer
import faiss, numpy as np, random, os

# === Config ===
app = Flask(__name__)
LINE_CHANNEL_ACCESS_TOKEN = os.getenv("LINE_ACCESS_TOKEN", "<YOUR_TOKEN>")
LINE_CHANNEL_SECRET = os.getenv("LINE_SECRET", "<YOUR_SECRET>")
line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# === Neo4j ===
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

# === Embedding Model ===
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# === เตรียม Keywords จาก Neo4j ===
def load_keywords():
    with driver.session() as session:
        query = """
        MATCH (p:Plant)-[:BELONGS_TO]->(c:Category)
        RETURN p.name AS plant, c.name AS category
        """
        result = session.run(query)
        data = []
        for record in result:
            data.append(record["plant"])
            data.append(record["category"])
        return list(set(data))  # ลบซ้ำ

keywords = load_keywords()
embeddings = model.encode(keywords, convert_to_numpy=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# === FAISS search ===
def find_similar_keyword(user_text):
    vec = model.encode([user_text], convert_to_numpy=True)
    D, I = index.search(vec, 1)
    return keywords[I[0][0]]

# === Query Neo4j ===
def search_plants(keyword):
    with driver.session() as session:
        query = """
        MATCH (p:Plant)-[:BELONGS_TO]->(c:Category)
        WHERE p.name CONTAINS $kw OR c.name CONTAINS $kw
        RETURN p.name AS name, p.price AS price, c.name AS category, p.image AS image
        LIMIT 5
        """
        result = session.run(query, kw=keyword)
        return [record.data() for record in result]

# === Random Example Questions ===
def random_questions():
    qlist = [
        "มีต้นไม้ราคาต่ำกว่า 300 ไหม",
        "ต้นไม้ที่นิยมทำรั้วคืออะไรบ้าง",
        "แนะนำต้นไม้ตระกูลสนราคาไม่เกิน 500",
        "ไม้ล้อมที่เหมาะกับบ้านสวนคืออะไร",
        "อยากได้ต้นไม้ที่มีกลิ่นหอม",
        "มีรูปตัวอย่างต้นไม้พร้อมราคาไหม",
        "ต้นไม้สำหรับตกแต่งสไตล์อังกฤษมีอะไรบ้าง",
        "ช่วยเลือกต้นไม้ที่โตเร็วและบังแดดได้ดี",
        "มีต้นไม้สูงใหญ่ราคาไม่เกิน 2000 ไหม",
        "แนะนำต้นไม้ที่ออกดอกสวย"
    ]
    return random.sample(qlist, k=random.randint(6, 12))

# === LINE Webhook ===
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'

# === LINE Event ===
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_text = event.message.text.strip()

    # หา keyword ใกล้เคียง
    similar_kw = find_similar_keyword(user_text)

    # Query Neo4j
    plants = search_plants(similar_kw)

    if plants:
        reply = f"🔍 คุณค้นหาใกล้เคียง: {similar_kw}\n\n"
        for p in plants:
            reply += f"- {p['name']} ({p['category']}) ราคา {p['price']} บาท\n"
    else:
        reply = "ไม่พบข้อมูลที่ตรงกับคำค้นหา"

    # แนบคำถามสุ่มบางข้อ
    reply += "\n\n❓ ตัวอย่างคำถามที่คุณลองได้:\n"
    for q in random_questions():
        reply += f"- {q}\n"

    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=reply)
    )

if __name__ == "__main__":
    app.run(port=5000, debug=True)
