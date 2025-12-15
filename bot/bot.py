import os
import discord
from vector_store import create_faiss_store
from rag_chain import load_llm, generate_answer

TOKEN = "xxxxxxxxx"
FAQ_PATH = "faq_dataset.txt"

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# Load components once
vector_store = create_faiss_store(FAQ_PATH)
text_gen_pipeline = load_llm()

@client.event
async def on_ready():
    print(f"Logged in as {client.user}")

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    user_query = message.content

    docs = vector_store.similarity_search(user_query, k=2)
    if not docs:
        await message.channel.send("Sorry, I couldn't find relevant information.")
        return

    context = "\n\n".join(doc.page_content for doc in docs)

    answer = generate_answer(text_gen_pipeline, context, user_query)
    await message.channel.send(answer)

client.run(TOKEN)
