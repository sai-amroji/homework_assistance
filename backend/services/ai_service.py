import sympy as sp
import re
import wikipedia
import torch

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

# ---------------------------
# MODEL SETUP
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "google/flan-t5-small"

tokenizer = None
model = None

def load_main_model():
    global tokenizer, model
    try:
        if tokenizer is None or model is None:
            print("🔄 Loading AI model...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
            model.eval()
            print("✅ Model loaded")
    except Exception as e:
        print("❌ Model load failed:", e)
        tokenizer = None
        model = None

# ---------------------------
# TEXT GENERATION
# ---------------------------
def generate_text(prompt: str, max_new_tokens: int = 150):
    try:
        load_main_model()  # ✅ lazy load

        # fallback if model not available
        if tokenizer is None or model is None:
            return None

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.5,
            do_sample=True,
        )

        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    except Exception as e:
        print("AI Error:", e)
        return None

# ---------------------------
# MAIN ANSWER FUNCTION
# ---------------------------
def generate_answer(question: str, session_id="default", subject="General"):

    try:
        prompt = f"Answer clearly in simple terms:\n\nQuestion: {question}\nAnswer:"

        answer = generate_text(prompt)

        # ✅ fallback if model fails (important)
        if not answer or len(answer.strip()) < 5:
            wiki = wikipedia.summary(question, sentences=2)
            if wiki:
                return f"📘 Answer:\n\n{wiki}"

            return f"📘 Answer:\n\nSorry, I couldn't generate a detailed answer. Try rephrasing your question."

        return f"📘 Answer:\n\n{answer}"

    except Exception as e:
        print("Main Error:", e)
        return "❌ Error generating answer"