from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def load_llm():
    model_name = "distilgpt2"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    text_gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        temperature=0.3,
        do_sample=True
    )

    return text_gen_pipeline


def generate_answer(text_gen_pipeline, context, question):
    prompt = f"""
You are a helpful customer support assistant.
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    response = text_gen_pipeline(prompt)[0]["generated_text"]

    # Return only the answer part
    return response.split("Answer:")[-1].strip()
