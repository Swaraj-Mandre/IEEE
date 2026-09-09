import json
import re

from llama_cpp import Llama, LlamaGrammar

from src import config

# Forces the model to emit a JSON array of 1-4 {concept, prerequisite} objects.
# Invalid tokens are masked during sampling, so malformed output cannot happen.
# An empty array is deliberately not allowed: given the choice, Phi-3 returns []
# for every chunk. Ungrounded triples are dropped afterwards instead.
TRIPLE_GRAMMAR = r"""
root   ::= "[" ws pair (ws "," ws pair)? (ws "," ws pair)? (ws "," ws pair)? ws "]"
pair   ::= "{" ws "\"concept\"" ws ":" ws string ws "," ws "\"prerequisite\"" ws ":" ws string ws "}"
string ::= "\"" ([^"\\\x7F\x00-\x1F])+ "\""
ws     ::= | " " | "\n" [ \t]*
"""

EXTRACTION_PROMPT = """You are an expert STEM educator analyzing a textbook passage.
Your task is to extract concept-prerequisite relationships from the given text.

Rules:
1. A "concept" is a key STEM idea explicitly mentioned in the passage.
2. A "prerequisite" is another concept that must be understood BEFORE the concept.
3. Both must be short noun phrases of 1 to 3 words, not sentences.
4. Both must appear in or be directly implied by the passage.
5. Extract 1 to 4 pairs maximum. Do not invent concepts not in the text.
6. If no clear prerequisite relationship exists, output an empty array.

Output format:
[
  {{"concept": "acceleration", "prerequisite": "velocity"}},
  {{"concept": "velocity", "prerequisite": "speed"}}
]
{section_hint}
Passage:
{chunk_text}

JSON output:"""


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


def load_grammar() -> LlamaGrammar:
    """Compile the extraction grammar.

    from_string() prints parse errors instead of raising and hands back a broken
    object that crashes the sampler later, so the result is checked here.
    """
    grammar = LlamaGrammar.from_string(TRIPLE_GRAMMAR, verbose=False)
    if not getattr(grammar, "_grammar", None):
        raise RuntimeError("Extraction grammar failed to compile - check TRIPLE_GRAMMAR syntax.")
    return grammar


def is_grounded(phrase: str, chunk_text: str) -> bool:
    """True if the phrase, or all of its words, appear in the passage."""
    text = chunk_text.lower()
    phrase = phrase.lower()
    if phrase in text:
        return True
    return all(word in text for word in phrase.split())


def build_prompt(chunk: dict) -> str:
    section = chunk.get("section")
    hint = f"\nThe passage is from the section \"{section}\".\n" if section else "\n"
    return EXTRACTION_PROMPT.format(section_hint=hint, chunk_text=chunk["text"])


def extract_triples_from_chunk(
    model: Llama,
    chunk: dict,
    grammar: LlamaGrammar,
    max_retries: int = 2,
) -> list[dict]:
    """
    Extract concept-prerequisite triples from a single chunk.

    Retries on failure and returns an empty list if every attempt fails,
    so one bad chunk never stops the pipeline.
    """
    prompt = build_prompt(chunk)

    for attempt in range(max_retries + 1):
        try:
            response = model(
                prompt,
                max_tokens=256,
                temperature=0.1,
                grammar=grammar,
                echo=False,
            )

            raw_output = response["choices"][0]["text"].strip()
            triples = parse_and_validate_output(raw_output, chunk["chunk_id"], chunk["text"])

            if triples is not None:
                # Attach chunk metadata to each triple
                for t in triples:
                    t["chunk_id"] = chunk["chunk_id"]
                    t["source_file"] = chunk["source_file"]
                    t["page_num"] = chunk["page_num"]
                    t["section"] = chunk.get("section")
                return triples

        except Exception as e:
            if attempt < max_retries:
                print(f"    Retry {attempt + 1} for chunk {chunk['chunk_id']}: {e}")
            else:
                print(f"    [FAILED] chunk {chunk['chunk_id']} after {max_retries} retries: {e}")

    return []


def parse_and_validate_output(raw: str, chunk_id: str, chunk_text: str) -> list[dict] | None:
    """
    Parse model output and keep only well-formed triples.

    Returns:
        List of valid triples, or None if parsing fails.
    """
    if not raw or raw.strip() == "":
        return []

    match = re.search(r"\[.*\]", raw, re.DOTALL)
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

    valid_triples = []
    for item in data:
        if not isinstance(item, dict):
            continue

        # Phi-3 sometimes returns a JSON null, so coerce before stripping.
        concept = (item.get("concept") or "").strip()
        prerequisite = (item.get("prerequisite") or "").strip()

        if not concept or not prerequisite:
            continue

        # Reject self-loops - a concept cannot be its own prerequisite
        if concept.lower() == prerequisite.lower():
            continue

        if len(concept) < 3 or len(prerequisite) < 3:
            continue

        # The grammar cannot express "no relationship here", so it forces a pair
        # out of every chunk. Anything not actually in the passage is invented.
        if not is_grounded(concept, chunk_text) or not is_grounded(prerequisite, chunk_text):
            continue

        valid_triples.append({
            "concept": concept,
            "prerequisite": prerequisite,
        })

    return valid_triples[:4]  # Hard cap at 4 triples per chunk
