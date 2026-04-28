import os, json, re, logging
logger = logging.getLogger(__name__)

def init():
    global model, tokenizer, wiki
    from mlx_lm import load
    model_dir = os.getenv("AZUREML_MODEL_DIR", ".")
    model, tokenizer = load(os.path.join(model_dir, "base_model"), adapter_path=os.path.join(model_dir, "adapters"))
    wiki_path = os.path.join(model_dir, "wiki_compiled.md")
    wiki = open(wiki_path).read() if os.path.exists(wiki_path) else ""
    logger.info("RIBO agent loaded")

def run(raw_data):
    data = json.loads(raw_data)
    question = data["question"]
    options = data.get("options", {})
    agent = data.get("agent", "v9_qlora")
    prompt = f"Question: {question}\n"
    for letter in ["A","B","C","D"]:
        if letter in options: prompt += f"{letter}. {options[letter]}\n"
    if agent == "v2_rewrite_wiki" and wiki:
        prompt = f"STUDY WIKI:\n{wiki[:25000]}\n\n{prompt}\nThink step by step. Cite the regulation. Answer with Final Answer: A, B, C, or D."
    elif agent == "v5_elimination":
        prompt += "\nWhich option is DEFINITELY wrong? Eliminate it, then the next worst, then pick. State Final Answer: A, B, C, or D."
    else:
        prompt += "\nThink step by step, then state Final Answer: A, B, C, or D."
    from mlx_lm import generate
    response = generate(model, tokenizer, prompt=prompt, max_tokens=400)
    if "<|endoftext|>" in response: response = response[:response.index("<|endoftext|>")]
    m = re.search(r"Final [Aa]nswer:\s*([A-D])", response)
    if not m: m = re.search(r"answer is\s*([A-D])", response, re.IGNORECASE)
    if not m: m = re.search(r"([A-D])\s*\.?\s*$", response.strip())
    answer = m.group(1).upper() if m else None
    return json.dumps({"answer": answer, "reasoning": response[:500], "agent": agent, "model": "qwen-2.5-7b-qlora-ribo-v1"})
