import os
import json
import re
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from openai import OpenAI

# Lokal geliştirme için .env destek (prod'da env panelinden verirsin)
load_dotenv()

app = Flask(__name__, static_folder="web", static_url_path="")

SYSTEM = """You are an expert X (Twitter) feed/ranking optimizer.

You must rewrite the user's draft to maximize predicted engagement probabilities
(reply, repost/retweet, like, click, dwell, share) WITHOUT changing facts.

CRITICAL RULES:
- Preserve meaning and factual claims exactly. Do NOT invent information.
- Do NOT forcefully shorten unless user asks.
- Keep natural Turkish crypto analyst tone.
- Avoid spam signals: excessive CAPS, excessive punctuation, aggressive CTA, link-bait.
- Output must be usable as-is for X (long flow is allowed, up to ~2000 chars).
"""

PROMPT = """USER DRAFT:
{draft}

TASK:
1) List issues that reduce engagement probability (bullet list).
2) Produce an optimized version that keeps the SAME facts and meaning.
   - Make it more readable (line breaks ok).
   - Increase reply probability: add a respectful tension / contrast point.
   - Increase repost probability: add one clear, confident takeaway.
   - Increase dwell: improve flow + structure (mini sections).
3) Estimate probabilities as numbers 0..1 for:
   like, reply, retweet, click, dwell, share, profile_click, follow_author,
   not_interested, block, report (negatives should be low).
4) Compute a weighted score using THESE DEFAULT WEIGHTS (can be edited later):
   like 0.18
   reply 0.30
   retweet 0.22
   click 0.08
   dwell 0.12
   share 0.10
   negatives: not_interested -0.30, block -0.70, report -0.90

Return JSON ONLY with keys:
issues (array of strings)
optimized_text (string)
probs (object of floats)
"""

DEFAULT_WEIGHTS = {
    "like": 0.18,
    "reply": 0.30,
    "retweet": 0.22,
    "click": 0.08,
    "dwell": 0.12,
    "share": 0.10,
    "not_interested": -0.30,
    "block": -0.70,
    "report": -0.90,
}

def compute_score(probs, weights):
    """
    Weighted sum -> normalize to 0..100.
    We add a small offset so neutral isn't always 0.
    """
    s = 0.0
    breakdown = {}
    for k, w in weights.items():
        v = float(probs.get(k, 0.0) or 0.0)
        breakdown[k] = v * float(w)
        s += breakdown[k]

    # simple normalization/clamp
    s = max(0.0, min(1.0, (s + 0.25)))
    return round(s * 100, 2), breakdown


def safe_extract_json(raw_text: str):
    """
    LLM bazen JSON dışında açıklama ekleyebiliyor.
    İlk { ... } bloğunu yakalayıp parse etmeye çalışıyoruz.
    """
    raw = (raw_text or "").strip()

    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return None, {"error": "LLM JSON döndürmedi", "raw": raw}

    candidate = m.group(0)
    try:
        return json.loads(candidate), None
    except Exception as e:
        return None, {"error": "JSON parse failed", "detail": str(e), "raw": raw}


@app.get("/")
def index():
    return send_from_directory("web", "index.html")


@app.post("/optimize")
def optimize():
    data = request.get_json(force=True) or {}
    draft = (data.get("draft") or "").strip()
    weights = data.get("weights") or DEFAULT_WEIGHTS

    if not draft:
        return jsonify({"error": "draft empty"}), 400

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return jsonify({"error": "OPENAI_API_KEY env yok. Hosting panelinde OPENAI_API_KEY ekle."}), 500

    client = OpenAI(api_key=api_key)

    msg = PROMPT.format(draft=draft)

    try:
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": msg},
            ],
            temperature=0.4,
        )
    except Exception as e:
        return jsonify({"error": "OpenAI request failed", "detail": str(e)}), 500

    raw = (resp.choices[0].message.content or "").strip()

    obj, err = safe_extract_json(raw)
    if err:
        return jsonify(err), 500

    issues = obj.get("issues", [])
    optimized_text = obj.get("optimized_text", "")
    probs = obj.get("probs", {})

    clean_probs = {}
    if isinstance(probs, dict):
        for k, v in probs.items():
            try:
                clean_probs[k] = float(v)
            except Exception:
                clean_probs[k] = 0.0

    score, breakdown = compute_score(clean_probs, weights)

    obj["issues"] = issues if isinstance(issues, list) else [str(issues)]
    obj["optimized_text"] = optimized_text if isinstance(optimized_text, str) else str(optimized_text)
    obj["probs"] = clean_probs
    obj["score"] = score
    obj["score_breakdown"] = breakdown

    return jsonify(obj)


if __name__ == "__main__":
    # ✅ Deploy uyumlu: 0.0.0.0 + PORT env
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
