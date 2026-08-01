import os
import csv
import json
import io
import requests

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# ============================================================
# CONFIGURATION
# ============================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Primary models and fallbacks for each exam question.
#
# OpenRouter automatically tries the fallback models if the
# primary model becomes unavailable/rate-limited/etc.
#
# Q1 = GPT
# Q2 = Gemini
# Q3 = Claude
# Q4 = GPT

MODEL_CONFIG = {
    "q1": {
        "name": "GPT-5.3 Chat",
        "primary": "openai/gpt-5.3-chat",
        "fallbacks": [
            "anthropic/claude-sonnet-4.6",
            "google/gemini-3.6-flash",
        ],
        "max_tokens": 3000,
    },

    "q2": {
        "name": "Gemini 3.6 Flash",
        "primary": "google/gemini-3.6-flash",
        "fallbacks": [
            "anthropic/claude-sonnet-4.6",
            "openai/gpt-5.3-chat",
        ],
        "max_tokens": 4000,
    },

    "q3": {
        "name": "Claude Sonnet 4.6",
        "primary": "anthropic/claude-sonnet-4.6",
        "fallbacks": [
            "google/gemini-3.6-flash",
            "openai/gpt-5.3-chat",
        ],
        "max_tokens": 4500,
    },

    "q4": {
        "name": "GPT-5.3 Chat",
        "primary": "openai/gpt-5.3-chat",
        "fallbacks": [
            "anthropic/claude-sonnet-4.6",
            "google/gemini-3.6-flash",
        ],
        "max_tokens": 600,
    },
}


# ============================================================
# EXAM-SPECIFIC PROMPTS
# ============================================================

Q1_PROMPT = """
You are assisting with Question 1 of an open-internet journalism/AI
evaluation exam.

TASK:
Design a MODEL-AGNOSTIC, REUSABLE rubric for evaluating whether an
article exhibits signs commonly associated with LLM-generated writing.

IMPORTANT:
- Do NOT claim that stylistic signals can prove that text was written by AI.
- Clearly distinguish "AI-like signals" from actual proof of AI authorship.
- The rubric should work across different LLMs.
- Avoid relying on specific phrases that only one model uses.
- Consider that humans can naturally exhibit many of these signals.
- Use multiple signals rather than a single detector.
- Include evidence, scoring, limitations, and an overall interpretation.

The examiner values:
1. Model-agnostic reasoning
2. Reusability
3. Specific and observable criteria
4. Nuance
5. Avoiding false certainty
6. Originality

Return a polished answer that could be adapted directly into an exam
submission.

If an article is supplied, evaluate THAT article as well.
"""


Q2_PROMPT = """
You are assisting with Question 2 of an investigative journalism/data
analysis exam.

The task is to identify THREE insights from the supplied dataset or API.

The three insights must be:
- impactful,
- practically relevant,
- genuinely surprising or non-obvious,
- supported by the supplied evidence.

Do NOT simply give obvious descriptive statistics.

For each insight provide:

1. INSIGHT
   State the finding clearly.

2. EVIDENCE
   Explain exactly what data supports it.
   Include relevant numbers, groups, comparisons, trends, anomalies,
   correlations, changes, or other evidence where available.

3. WHY IT MATTERS
   Explain the real-world or journalistic significance.

4. HOW A JOURNALIST COULD INDEPENDENTLY VERIFY IT
   Give concrete verification steps, such as:
   - contacting named stakeholders,
   - requesting records,
   - checking official databases,
   - comparing against another dataset,
   - inspecting original documents,
   - interviewing affected parties,
   - reproducing the calculation.

5. LIMITATIONS / ALTERNATIVE EXPLANATIONS
   Explain what the dataset does NOT establish and what could
   potentially explain the finding.

IMPORTANT:
- Do not invent data.
- Do not invent sources.
- Do not treat correlation as causation.
- If the supplied material is insufficient to support a claim,
  explicitly say so.
- Rank the three insights by journalistic importance.
- Prefer surprising findings over generic observations.

If calculations are possible from the supplied data, perform them
carefully and show enough detail for the user to verify them.
"""


Q3_PROMPT = """
You are assisting with Question 3 of an investigative journalism exam.

The supplied material may contain reports, emails, financial records,
operational information, spreadsheets, notes, or other organizational
documents.

Your job is to analyze the material and produce TWO things:

PART A — INTERVIEW QUESTIONS

Identify the most important unresolved issues, inconsistencies,
anomalies, unexplained decisions, financial questions, operational
risks, or contradictions.

Then produce high-value interview questions that could deepen an
investigation.

Questions should:
- be specific to the supplied evidence,
- identify who should be asked where possible,
- reference the relevant issue,
- avoid generic questions,
- seek facts rather than merely opinions,
- help uncover accountability, causation, or missing information.

Prioritize the questions rather than producing a huge generic list.

PART B — HIGH-VALUE AI USE CASES

Identify practical AI use cases that would genuinely help this
organization.

For each use case explain:
- the problem,
- how AI could help,
- what data/material it would use,
- expected benefit,
- important risks or limitations,
- how a human should remain involved.

Do NOT recommend AI simply because it sounds modern.

IMPORTANT:
- Do not invent facts not present in the material.
- Distinguish evidence from inference.
- Highlight contradictions and uncertainty.
- Think like an investigative journalist rather than a generic business
  consultant.
"""


Q4_PROMPT = """
You are assisting with Question 4 of an exam requiring a SHORT ESSAY.

The final answer MUST be no more than 150 words.

The examiner wants:
- a direct response to the assigned topic,
- an original perspective,
- a distinctive argument or framing,
- concrete reasoning,
- concise and natural writing.

Avoid:
- generic AI-style introductions,
- clichés,
- unnecessary headings,
- excessive "on the one hand / on the other hand" balance,
- repetitive conclusions,
- empty claims such as "technology is changing the world",
- unnecessarily sophisticated vocabulary.

Before writing the final essay, internally consider several possible
angles and choose the least generic and most defensible one.

Return ONLY the final essay followed by:

Word count: X

The essay itself must be <=150 words.
"""


# ============================================================
# HELPERS
# ============================================================

def get_prompt(question_type):
    prompts = {
        "q1": Q1_PROMPT,
        "q2": Q2_PROMPT,
        "q3": Q3_PROMPT,
        "q4": Q4_PROMPT,
    }
    return prompts.get(question_type)


def clean_text(value):
    if value is None:
        return ""

    if not isinstance(value, str):
        value = str(value)

    # Avoid accidentally sending enormous repeated whitespace.
    return value.strip()


def extract_file_text(file_storage):
    """
    Reads common exam material formats without requiring pandas.

    Supported:
    - txt
    - csv
    - json
    - md
    - html
    """

    if not file_storage:
        return ""

    filename = (file_storage.filename or "").lower()
    raw = file_storage.read()

    # Try UTF-8 first.
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1", errors="replace")

    if filename.endswith(".json"):
        try:
            parsed = json.loads(text)
            return json.dumps(parsed, indent=2, ensure_ascii=False)
        except Exception:
            return text

    if filename.endswith(".csv"):
        try:
            rows = list(csv.reader(io.StringIO(text)))
            output = io.StringIO()

            writer = csv.writer(output)
            writer.writerows(rows)

            return output.getvalue()
        except Exception:
            return text

    return text


def build_user_message(question, material):
    return f"""
EXAM QUESTION:

{question}

SUPPLIED MATERIAL / DATA:

{material}

Analyze only the information provided above plus general reasoning.
Do not fabricate facts, numbers, sources, people, or documents.
"""


def call_openrouter(question_type, question, material):
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not configured in Render."
        )

    config = MODEL_CONFIG[question_type]

    primary = config["primary"]
    fallbacks = config["fallbacks"]

    # OpenRouter model fallback list.
    #
    # Primary model is specified in "model".
    # Remaining models are fallback candidates.
    models = [primary] + fallbacks

    system_prompt = get_prompt(question_type)

    user_message = build_user_message(
        question=question,
        material=material,
    )

    payload = {
        "model": primary,

        # OpenRouter model-level fallback support.
        "models": models,

        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_message,
            },
        ],

        "max_tokens": config["max_tokens"],

        # Keep provider failover enabled.
        "provider": {
            "allow_fallbacks": True
        },
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",

        # Optional OpenRouter metadata.
        "HTTP-Referer": "https://roe-w9pc.onrender.com",
        "X-Title": "ROE Exam Assistant",
    }

    response = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=180,
    )

    # Handle errors with useful messages.
    if response.status_code != 200:
        try:
            error_data = response.json()
        except Exception:
            error_data = response.text

        raise RuntimeError(
            f"OpenRouter HTTP {response.status_code}: {error_data}"
        )

    data = response.json()

    if "choices" not in data or not data["choices"]:
        raise RuntimeError(
            f"OpenRouter returned no choices: {data}"
        )

    message = data["choices"][0].get("message", {})
    answer = message.get("content", "")

    if not answer:
        raise RuntimeError(
            f"OpenRouter returned an empty response: {data}"
        )

    # OpenRouter tells us which model actually answered.
    actual_model = data.get("model", primary)

    usage = data.get("usage", {})

    return {
        "answer": answer,
        "model": actual_model,
        "requested_model": primary,
        "usage": usage,
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "openrouter_key_configured": bool(OPENROUTER_API_KEY),
        "models": {
            key: value["primary"]
            for key, value in MODEL_CONFIG.items()
        },
    })


# ============================================================
# MAIN AI ENDPOINT
# ============================================================

@app.route("/api/ask", methods=["POST"])
def ask():
    try:
        # Supports both JSON requests and multipart/form-data.
        if request.is_json:
            data = request.get_json(silent=True) or {}

            question_type = clean_text(
                data.get("question_type", "")
            ).lower()

            question = clean_text(
                data.get("question", "")
            )

            material = clean_text(
                data.get("material", "")
            )

        else:
            question_type = clean_text(
                request.form.get("question_type", "")
            ).lower()

            question = clean_text(
                request.form.get("question", "")
            )

            material = clean_text(
                request.form.get("material", "")
            )

            uploaded_file = request.files.get("file")

            if uploaded_file:
                file_text = extract_file_text(uploaded_file)

                if material and file_text:
                    material = (
                        material
                        + "\n\n"
                        + "UPLOADED FILE:\n"
                        + file_text
                    )
                elif file_text:
                    material = file_text

        if question_type not in MODEL_CONFIG:
            return jsonify({
                "error": "Please select Q1, Q2, Q3, or Q4."
            }), 400

        if not question:
            return jsonify({
                "error": "Please enter the exam question."
            }), 400

        # Q1/Q2/Q3 can work with material, but Q4 might not need it.
        if not material:
            material = "(No additional material supplied.)"

        result = call_openrouter(
            question_type=question_type,
            question=question,
            material=material,
        )

        return jsonify({
            "success": True,
            "question_type": question_type,
            "answer": result["answer"],
            "model": result["model"],
            "requested_model": result["requested_model"],
            "usage": result["usage"],
        })

    except requests.exceptions.Timeout:
        return jsonify({
            "error": (
                "The model took too long to respond. "
                "Try again once. If the problem continues, "
                "the provider may be busy."
            )
        }), 504

    except requests.exceptions.RequestException as e:
        return jsonify({
            "error": f"Network error connecting to OpenRouter: {str(e)}"
        }), 502

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# SIMPLE FRONTEND
# ============================================================

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <title>ROE Exam Assistant</title>

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            color: #202124;
        }

        .container {
            max-width: 1100px;
            margin: 30px auto;
            padding: 20px;
        }

        .card {
            background: white;
            border-radius: 14px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 4px 18px rgba(0,0,0,0.08);
        }

        h1 {
            margin-top: 0;
        }

        .subtitle {
            color: #666;
            margin-bottom: 25px;
        }

        label {
            display: block;
            font-weight: bold;
            margin-top: 18px;
            margin-bottom: 7px;
        }

        select,
        textarea,
        input[type="file"] {
            width: 100%;
            border: 1px solid #ccc;
            border-radius: 8px;
            padding: 12px;
            font-size: 15px;
        }

        textarea {
            min-height: 180px;
            resize: vertical;
            font-family: Arial, sans-serif;
        }

        #material {
            min-height: 260px;
        }

        button {
            margin-top: 20px;
            border: none;
            border-radius: 8px;
            padding: 13px 22px;
            font-size: 16px;
            font-weight: bold;
            cursor: pointer;
            background: #111827;
            color: white;
        }

        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .model-box {
            margin-top: 15px;
            padding: 12px;
            border-radius: 8px;
            background: #eef3ff;
            font-size: 14px;
        }

        .answer {
            white-space: pre-wrap;
            line-height: 1.6;
            font-size: 16px;
        }

        .status {
            margin-top: 15px;
            color: #555;
        }

        .error {
            background: #fff0f0;
            border: 1px solid #ffcccc;
            color: #a00000;
            padding: 15px;
            border-radius: 8px;
            white-space: pre-wrap;
        }

        .tips {
            background: #f7f7f7;
            padding: 15px;
            border-radius: 8px;
            line-height: 1.5;
        }

        .small {
            color: #777;
            font-size: 13px;
        }
    </style>
</head>

<body>

<div class="container">

    <div class="card">
        <h1>ROE Exam Assistant</h1>

        <div class="subtitle">
            OpenRouter-powered assistant for Questions 1–4.
        </div>

        <div class="tips">
            <strong>Exam workflow:</strong>
            Select the question type, paste the exam question,
            then paste the relevant dataset/documents/API output.
            For Q4, the material field can be left empty.
        </div>
    </div>


    <div class="card">

        <label for="questionType">
            Exam Question
        </label>

        <select id="questionType">
            <option value="q1">
                Q1 — LLM Verification Rubric
            </option>

            <option value="q2">
                Q2 — Dataset/API Insights
            </option>

            <option value="q3">
                Q3 — Organization Analysis
            </option>

            <option value="q4">
                Q4 — Short Essay
            </option>
        </select>


        <div id="modelInfo" class="model-box">
            Primary model:
            <strong>GPT-5.3 Chat</strong>
        </div>


        <label for="question">
            Exam Question
        </label>

        <textarea
            id="question"
            placeholder="Paste the exact exam question here..."
        ></textarea>


        <label for="material">
            Dataset / Article / Documents / API Output
        </label>

        <textarea
            id="material"
            placeholder="Paste the relevant material here. For Q2/Q3 this is where you can paste dataset/API output, reports, emails, financial records, etc."
        ></textarea>


        <label for="file">
            Optional text/CSV/JSON file
        </label>

        <input
            type="file"
            id="file"
            accept=".txt,.csv,.json,.md,.html"
        >

        <div class="small">
            The selected file is read by your browser and sent to the
            backend as text. Do not upload passwords, API keys, or
            unrelated private information.
        </div>


        <button id="generateBtn" onclick="generateAnswer()">
            Generate Answer
        </button>

        <div id="status" class="status"></div>

    </div>


    <div class="card" id="resultCard" style="display:none;">

        <h2>AI Response</h2>

        <div id="resultModel" class="model-box"></div>

        <br>

        <div id="answer" class="answer"></div>

    </div>

</div>


<script>

const MODEL_NAMES = {
    q1: "GPT-5.3 Chat",
    q2: "Gemini 3.6 Flash",
    q3: "Claude Sonnet 4.6",
    q4: "GPT-5.3 Chat"
};


const questionType = document.getElementById("questionType");

questionType.addEventListener("change", function() {

    const type = this.value;

    document.getElementById("modelInfo").innerHTML =
        "Primary model: <strong>" +
        MODEL_NAMES[type] +
        "</strong>";
});


async function generateAnswer() {

    const button =
        document.getElementById("generateBtn");

    const status =
        document.getElementById("status");

    const resultCard =
        document.getElementById("resultCard");

    const answerBox =
        document.getElementById("answer");

    const resultModel =
        document.getElementById("resultModel");


    const type =
        document.getElementById("questionType").value;

    const question =
        document.getElementById("question").value.trim();

    const material =
        document.getElementById("material").value.trim();

    const fileInput =
        document.getElementById("file");


    if (!question) {
        alert("Please paste the exam question first.");
        return;
    }


    button.disabled = true;

    button.innerText = "Analyzing...";

    status.innerText =
        "Sending the question to OpenRouter. Please wait...";

    resultCard.style.display = "none";


    try {

        let finalMaterial = material;


        // Read optional local file in the browser.
        if (fileInput.files.length > 0) {

            const file = fileInput.files[0];

            const fileText =
                await file.text();

            if (finalMaterial) {

                finalMaterial +=
                    "\\n\\n--- UPLOADED FILE: " +
                    file.name +
                    " ---\\n" +
                    fileText;

            } else {

                finalMaterial =
                    "--- UPLOADED FILE: " +
                    file.name +
                    " ---\\n" +
                    fileText;
            }
        }


        const response =
            await fetch("/api/ask", {

                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({

                    question_type: type,

                    question: question,

                    material: finalMaterial
                })
            });


        const data =
            await response.json();


        if (!response.ok || !data.success) {

            throw new Error(
                data.error ||
                "Unknown server error."
            );
        }


        resultCard.style.display = "block";


        answerBox.innerText =
            data.answer;


        resultModel.innerHTML =
            "<strong>Model requested:</strong> " +
            data.requested_model +
            "<br>" +
            "<strong>Model that answered:</strong> " +
            data.model;


        if (data.usage) {

            const usageText =
                "<br><br><strong>Token usage:</strong> " +
                "Input: " +
                (data.usage.prompt_tokens ?? "N/A") +
                " | Output: " +
                (data.usage.completion_tokens ?? "N/A");

            resultModel.innerHTML += usageText;
        }


        status.innerText =
            "Completed successfully.";


    } catch (error) {

        resultCard.style.display = "block";

        answerBox.innerHTML =
            '<div class="error">' +
            escapeHtml(error.message) +
            '</div>';

        resultModel.innerHTML = "";

        status.innerText =
            "Request failed.";


    } finally {

        button.disabled = false;

        button.innerText =
            "Generate Answer";
    }
}


function escapeHtml(text) {

    const div =
        document.createElement("div");

    div.innerText = text;

    return div.innerHTML;
}

</script>

</body>
</html>
"""


# ============================================================
# FRONT PAGE
# ============================================================

@app.route("/", methods=["GET"])
def index():
    return render_template_string(HTML)


# ============================================================
# RENDER ENTRY POINT
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )
    
