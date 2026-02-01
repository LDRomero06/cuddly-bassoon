import numpy as np
import os

from js import window, Object, Reflect  # <--- Added Reflect
from pyodide.ffi import to_js
import asyncio

OLLAMA_HOST=os.getenv("OLLAMA_HOST")

class RAGEngine:
    def __init__(self):
        self.chunks = []
        self.vectors = None
        self.is_ready = False

    async def initialize(self):
        """Waits for the JS models to load."""
        while not window.models.embedder or not window.models.generator:
            await asyncio.sleep(1)
        self.is_ready = True

    def recursive_chunker(self, text, max_size=400):
        raw_paragraphs = text.split("\n\n")
        chunks = []
        for p in raw_paragraphs:
            if not p.strip(): continue
            if len(p) > max_size:
                for i in range(0, len(p), max_size):
                    chunks.append(p[i:i+max_size])
            else:
                chunks.append(p)
        self.chunks = chunks
        return len(self.chunks)

    async def index_documents(self):
        """Embeds all chunks currently in self.chunks."""
        if not self.chunks:
            return
            
        vectors = []
        options = {"pooling": "mean", "normalize": True}
        # Safely convert Python dict to JS Object
        js_options = Object.fromEntries(to_js(options.items()))
        
        for chunk in self.chunks:
            # Use Reflect.apply(target_function, this_context, argument_list)
            # This is the most reliable way to call JS pipelines from Pyodide
            js_output = await Reflect.apply(window.models.embedder, None, to_js([chunk, js_options]))
            
            v = np.array(js_output.data.to_py()) 
            vectors.append(v)
        
        self.vectors = np.vstack(vectors)
        print(f"Indexed {len(self.chunks)} chunks.")

    async def generate_response(self, user_query):
        context_text = ""
        
        # 1. Retrieval (Keep this as is)
        if self.vectors is not None:
            options = {"pooling": "mean", "normalize": True}
            js_options = Object.fromEntries(to_js(options.items()))
            q_out = await Reflect.apply(window.models.embedder, None, to_js([user_query, js_options]))
            q_vec = np.array(q_out.data.to_py())
            similarities = np.dot(self.vectors, q_vec)
            top_indices = np.argsort(similarities)[-2:][::-1]
            found_chunks = [self.chunks[i] for i in top_indices]
            context_text = "\n".join(found_chunks)
            

        # 2. MANUAL FORMATTING (The fix for "e is null")
        # We wrap the text in Gemma's specific turn markers
        window.console.log("Context:",context_text)
        window.console.log("User query:", user_query)
        full_prompt = (
            f"<start_of_turn>user\n"
            f"Use the following context to answer the question.\n"
            f"Context: {context_text}\n\n"
            f"Question: {user_query}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        
        gen_params = {
            "max_new_tokens": 150, 
            "temperature": 0.5, 
            "do_sample": False,
            "repetition_penalty": 1.2, # <--- CRITICAL: Penalizes repeated tokens
            "repetition_range": 50,
            "return_full_text": True,
        }
        js_gen_params = Object.fromEntries(to_js(gen_params.items()))

        # 3. Pass the string directly instead of a message list
        # This bypasses the template engine entirely
        result = await Reflect.apply(window.models.generator, None, to_js([full_prompt, js_gen_params]))
        
        # 4. Safe extraction
        try:
            # Result for a string input is usually [{generated_text: "..."}]
            output = result.to_py()
            return output[0]['generated_text']
        except:
            return str(result[0].generated_text)