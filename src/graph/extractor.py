import json
import re
from llama_cpp import Llama

from src import config


def load_model() -> Llama:
    """Load Phi-3 Mini for triple extraction. Set N_GPU_LAYERS=-1 in .env to use a GPU."""
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {config.MODEL_PATH}\n"
            f"Update MODEL_PATH in your .env file.\n"
            f"Download from: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf"
        )

    print(f"Loading model from: {config.MODEL_PATH}")
    print("This takes 10-30 seconds on first load...")

    model = Llama(
        model_path=str(config.MODEL_PATH),
        n_ctx=config.N_CTX,
        n_gpu_layers=config.N_GPU_LAYERS,
        n_threads=config.N_THREADS,
        verbose=False,
    )

    print("Model loaded successfully.\n")
    return model

EXTRACTION_PROMPT = """You are an expert STEM educator analyzing a textbook passage.
Your task is to extract concept-prerequisite relationships from the given text.

Rules:
1. A "concept" is a key STEM idea explicitly mentioned in the passage.
2. A "prerequisite" is another concept that must be understood BEFORE the concept.
3. Both concept and prerequisite must appear in or be directly implied by the passage.
4. Extract 1 to 4 pairs maximum. Do not invent concepts not in the text.
5. Output ONLY a JSON array. No explanation. No markdown. No extra text.

Output format:
[
  {{"concept": "concept name", "prerequisite": "prerequisite name"}},
  {{"concept": "concept name", "prerequisite": "prerequisite name"}}
]

If no clear prerequisite relationships exist in the passage, output:
[]

Passage:
{chunk_text}

JSON output:"""


def extract_triples_from_chunk(
    model: Llama,
    chunk: dict,
    max_retries: int = 2
) -> list[dict]:
    """
    Extract concept-prerequisite triples from a single chunk.

    Retries up to max_retries times if output is malformed.
    Returns empty list if all retries fail — never crashes the pipeline.
    """
    prompt = EXTRACTION_PROMPT.format(chunk_text=chunk["text"])

    for attempt in range(max_retries + 1):
        try:
            response = model(
                prompt,
                max_tokens=256,      # Enough for 4 triples in JSON
                temperature=0.1,     # Low temperature = more deterministic output
                stop=["Passage:", "Rules:", "\n\n\n"],  # Stop tokens
                echo=False,
            )

            raw_output = response["choices"][0]["text"].strip()

            # Parse and validate JSON
            triples = parse_and_validate_output(raw_output, chunk["chunk_id"])

            if triples is not None:
                # Attach chunk metadata to each triple
                for t in triples:
                    t["chunk_id"] = chunk["chunk_id"]
                    t["source_file"] = chunk["source_file"]
                    t["page_num"] = chunk["page_num"]
                return triples

        except Exception as e:
            if attempt < max_retries:
                print(f"    Retry {attempt + 1} for chunk {chunk['chunk_id']}: {e}")
            else:
                print(f"    [FAILED] chunk {chunk['chunk_id']} after {max_retries} retries: {e}")

    return []


def parse_and_validate_output(raw: str, chunk_id: str) -> list[dict] | None:
    """
    Parse SLM output and validate it matches expected structure.

    Returns:
        List of valid triples, or None if parsing fails.
    """
    if not raw or raw.strip() == "":
        return []

    # Sometimes the SLM wraps output in markdown code blocks
    # Strip them before parsing
    raw = re.sub(r"```json\s*", "", raw)
    raw = re.sub(r"```\s*", "", raw)
    raw = raw.strip()

    # Extract JSON array from output
    # SLM sometimes adds text before/after the array
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if not match:
        print(f"    [WARN] No JSON array found in output for chunk {chunk_id}")
        return None

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as e:
        print(f"    [WARN] JSON parse error for chunk {chunk_id}: {e}")
        return None

    if not isinstance(data, list):
        return None

    # Validate each triple has required fields with non-empty strings
    valid_triples = []
    for item in data:
        if not isinstance(item, dict):
            continue
        concept = item.get("concept", "").strip()
        prerequisite = item.get("prerequisite", "").strip()

        if not concept or not prerequisite:
            continue

        # Reject self-loops — a concept cannot be its own prerequisite
        if concept.lower() == prerequisite.lower():
            continue

        # Reject if concept and prerequisite are identical after normalization
        if len(concept) < 3 or len(prerequisite) < 3:
            continue

        valid_triples.append({
            "concept": concept,
            "prerequisite": prerequisite,
        })

    return valid_triples[:4]  # Hard cap at 4 triples per chunk