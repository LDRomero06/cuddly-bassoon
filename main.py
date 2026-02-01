from pyscript import document, window
from processor import RAGEngine
import asyncio

engine = RAGEngine()

async def startup():
    await window.initModels()
    await engine.initialize()
    document.getElementById("model-status").innerText = "Gemma 3 Online"
    document.querySelector(".dot").style.background = "#22c55e"

asyncio.ensure_future(startup())

def add_message(text, sender):
    history = document.getElementById("chat-history")
    div = document.createElement("div")
    div.className = f"message {sender}"
    meta = document.createElement("span")
    meta.className = "meta"
    meta.innerText = "You" if sender == "user" else "Gemma 3"
    content = document.createTextNode(text)
    div.appendChild(meta)
    div.appendChild(content)
    history.appendChild(div)
    history.scrollTop = history.scrollHeight

async def on_index_click(event):
    if not engine.is_ready:
        window.alert("Models loading...")
        return

    text = document.getElementById("doc-input").value
    if not text.strip(): return

    status_div = document.getElementById("index-status")
    status_div.innerText = "Chunking text..."
    
    count = engine.recursive_chunker(text)
    status_div.innerText = f"Embedding {count} chunks..."
    
    await engine.index_documents()
    status_div.innerText = f"✅ Indexed {count} chunks."

async def on_send_click(event):
    inp = document.getElementById("user-input")
    text = inp.value
    if not text.strip(): return
    
    add_message(text, "user")
    inp.value = ""
    status = document.getElementById("model-status")
    status.innerText = "Gemma is thinking..."
    
    try:
        response = await engine.generate_response(text)
        add_message(response, "ai")
    except Exception as e:
        add_message(f"Error: {str(e)}", "ai")
    
    status.innerText = "Gemma 3 Online"