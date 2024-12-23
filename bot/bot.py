import discord
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceLLM

# Discord Bot Setup
TOKEN = "MTMyMDY0ODQ3MDk1NjM0MzMxNg.GRk91v.HPoOOITMvFkSjx3BfvVfLORBjqNIMclVdaV0Mo"
intents = discord.Intents.default()
client = discord.Client(intents=intents)

# Load LangChain and FAISS
faq_data = load_faq_dataset("faq_dataset.txt")
faiss_store, faq_data = setup_faiss(faq_data)
llm = HuggingFaceLLM()

@client.event
async def on_ready():
    print(f"We have logged in as {client.user}")

@client.event
async def on_message(message):
    if message.author == client.user:
        return

    user_query = message.content
    relevant_docs = faiss_store.similarity_search(user_query, k=1)
    if relevant_docs:
        question_index = relevant_docs[0].metadata['index']
        answer = faq_data[question_index]["answer"]
        await message.channel.send(f"Answer: {answer}")
    else:
        await message.channel.send("Sorry, I couldn't find an answer to your question.")

client.run(TOKEN)
