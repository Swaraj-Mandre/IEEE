import os
from llama_cpp import Llama
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

SYSTEM_PROMPT = """You are Vidhya, a patient and encouraging STEM tutor for Class 9 students in rural India.
Your job is to explain one concept at a time in simple, clear language.

Rules:
1. Explain only the concept you are given. Do not jump ahead.
2. Use simple words. Avoid jargon unless you explain it immediately.
3. Use one real-life analogy or example from everyday Indian life.
4. Keep your explanation under 150 words.
5. End with one simple check question to test understanding.
6. Never say you are an AI."""


def load_model() -> Llama:
    model_path = os.getenv("MODEL_PATH")
    if not model_path or not Path(model_path).exists():
        raise FileNotFoundError(
            "Model not found at: " + str(model_path) +
            "\nUpdate MODEL_PATH in your .env file."
        )
    print("Loading model...")
    model = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=0,
        n_threads=4,
        verbose=False,
    )
    print("Model loaded.")
    return model


def build_prompt(concept: str, supporting_text: str, level: str = "simple") -> str:
    if level == "simple":
        style = "Use a very simple analogy. Explain like the student is hearing this for the first time."
    else:
        style = "Give a structured explanation with definition, example, and formula if applicable."

    prompt = SYSTEM_PROMPT
    prompt += "\n\nContext from textbook:\n" + supporting_text[:500]
    prompt += "\n\nConcept to explain: " + concept
    prompt += "\nStyle: " + style
    prompt += "\n\nYour explanation:"
    return prompt


def generate_explanation(
    model: Llama,
    concept: str,
    supporting_chunks: list,
    level: str = "normal",
) -> str:
    supporting_text = ""
    if supporting_chunks:
        supporting_text = " ".join([c["text"] for c in supporting_chunks[:2]])

    prompt = build_prompt(concept, supporting_text, level)

    response = model(
        prompt,
        max_tokens=300,
        temperature=0.7,
        stop=["Student:", "Question:", "\n\n\n"],
        echo=False,
    )

    explanation = response["choices"][0]["text"].strip()
    return explanation