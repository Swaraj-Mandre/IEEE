from llama_cpp import Llama

from src import config

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
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {config.MODEL_PATH}\n"
            "Update MODEL_PATH in your .env file."
        )
    print("Loading model...")
    model = Llama(
        model_path=str(config.MODEL_PATH),
        n_ctx=config.N_CTX,
        n_gpu_layers=config.N_GPU_LAYERS,
        n_threads=config.N_THREADS,
        verbose=False,
    )
    print("Model loaded.")
    return model


def build_prompt(concept: str, supporting_text: str, level: str = "simple") -> str:
    if level == "simple":
        style = "Use a very simple analogy. Explain like the student is hearing this for the first time."
    else:
        style = "Give a structured explanation with definition, example, and formula if applicable."

    user_content = SYSTEM_PROMPT
    user_content += "\n\nContext from textbook:\n" + supporting_text[:500]
    user_content += "\n\nConcept to explain: " + concept
    user_content += "\nStyle: " + style

    prompt = "<|user|>\n" + user_content + "<|end|>\n<|assistant|>\n"
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
    stop=["<|end|>", "<|assistant|>", "<|user|>", "Inquiry:", "Student:", "\n\n\n"],
    echo=False,
    )

    explanation = response["choices"][0]["text"].strip()
    return explanation