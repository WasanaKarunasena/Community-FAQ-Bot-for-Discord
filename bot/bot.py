import discord
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader


# Discord Bot Setup
TOKEN = ""  # Replace with your bot's token
intents = discord.Intents.default()
client = discord.Client(intents=intents)

# Load HuggingFace model and pipeline
model_name = "distilgpt2"  # Replace with a Hugging Face model suitable for text generation
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
hf_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Load FAQ dataset and FAISS
faq_data_path = r"C:\Users\wasana\Desktop\New folder (9)\bot\faq_dataset.txt"  # Full path to your FAQ dataset
loader = TextLoader(faq_data_path)

try:
    documents = loader.load()
    print("FAQ dataset loaded successfully.")
except Exception as e:
    print(f"Error loading FAQ dataset: {e}")
    exit(1)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

faiss_store = FAISS.from_documents(documents, embeddings)

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
        answer = relevant_docs[0].page_content
        await message.channel.send(f"Answer: {answer}")
    else:
        await message.channel.send("Sorry, I couldn't find an answer to your question.")

client.run(TOKEN)
