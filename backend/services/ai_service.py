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

# -----------------------------------------------------------------------
# LOAD AI MODEL
# -----------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "google/flan-t5-small"

tokenizer = None
model = None

def load_main_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        print("🔄 Loading AI model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME,
            low_cpu_mem_usage=True
        ).to(device)
        model.eval()
        print("✅ Model loaded successfully")

# ✅ IMPORTANT: LOAD MODEL ON START
load_main_model()

# -----------------------------------------------------------------------
# GENERATE TEXT
# -----------------------------------------------------------------------
def generate_text(prompt: str, max_new_tokens: int = 150) -> str | None:
    try:
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
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

# -----------------------------------------------------------------------
# SIMPLE ANSWER FUNCTION (MAIN FIX)
# -----------------------------------------------------------------------
def generate_answer(question: str, session_id: str = "default", subject: str = "General") -> str:
    
    try:
        prompt = f"Answer this clearly in simple terms:\n\nQuestion: {question}\nAnswer:"
        answer = generate_text(prompt)

        # ✅ fallback if model fails
        if not answer or len(answer.strip()) < 5:
            return f"⚠️ AI could not generate answer. Try again.\n\n(Question was: {question})"

        return answer

    except Exception as e:
        print("Main AI Error:", e)
        return "❌ Error generating answer"