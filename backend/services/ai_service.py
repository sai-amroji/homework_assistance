import sympy as sp
import re
import wikipedia

from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

# ---------------------------
# MATH SOLVER
# ---------------------------
def solve_math(question: str):
    try:
        q = question.replace("^", "**")
        transformations = standard_transformations + (implicit_multiplication_application,)
        expr = parse_expr(q, transformations=transformations)
        return f"📘 Answer:\n\n{expr}"
    except:
        return None

# ---------------------------
# MAIN ANSWER FUNCTION
# ---------------------------
def generate_answer(question: str, session_id="default", subject="General"):

    try:
        q = question.strip()

        # ✅ Math handling
        math_result = solve_math(q)
        if math_result:
            return math_result

        # ✅ Wikipedia answer
        try:
            wiki = wikipedia.summary(q, sentences=3)
            if wiki:
                return f"📘 Answer:\n\n{wiki}"
        except:
            pass

        # ✅ fallback
        return f"📘 Answer:\n\nYou asked: {q}\n\nTry asking in a clearer way."

    except Exception as e:
        print("Error:", e)
        return "❌ Error generating answer"