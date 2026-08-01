import os

from flask import Flask, render_template, request, jsonify
from openai import OpenAI


app = Flask(__name__)


def get_client():
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is not configured.")

    return OpenAI(
        api_key=api_key,
        base_url="https://aipipe.org/openrouter/v1"
    )


PROMPTS = {

    "q1": """
You are assisting with an open-internet examination question about
detecting common signs of AI-generated writing.

The task is to design/use a model-agnostic evaluation rubric.

Analyze the supplied question and material carefully.

The evaluation must NOT claim that a text is AI-generated merely
because it has a particular writing style. Treat the characteristics
as probabilistic signals rather than proof.

Consider signals such as:

- predictable or formulaic structure
- repetitive sentence patterns
- generic wording
- excessive hedging or artificial balance
- formulaic transitions
- lack of concrete details
- overly uniform tone
- repetitive summaries or conclusions
- generic examples
- unnatural consistency

For each relevant signal:
- explain what to look for
- identify evidence
- give a score from 0 to 3
- explain the reasoning

Then provide:
1. Overall assessment
2. Strongest evidence
3. Important limitations
4. How the rubric can be reused across different LLMs

Be careful not to overclaim.

EXAM QUESTION:
""",

    "q2": """
You are an investigative data journalist.

Analyze the supplied exam question and dataset/API material.

Identify EXACTLY THREE insights that are:

- impactful
- practically relevant
- genuinely surprising
- supported by evidence
- useful for further investigation

Do NOT simply report obvious averages, totals, or rankings.

Search for:
- unexpected patterns
- anomalies
- outliers
- group differences
- trends
- changes over time
- correlations
- counterintuitive results
- rare combinations
- ranking reversals
- possible confounding variables

For each of the three insights provide:

1. Insight title
2. What the data shows
3. Specific evidence
4. Why it matters
5. Why it is surprising
6. Possible explanation
7. How a journalist could independently verify it
8. Important limitations
9. Confidence level

Clearly distinguish correlation from causation.

If the supplied material is insufficient to support a claim,
say so rather than inventing evidence.

EXAM QUESTION:
""",

    "q3": """
You are an investigative journalist analyzing an organization.

Analyze the supplied exam question and organizational material.

Look for:

- contradictions
- missing information
- financial anomalies
- unexplained decisions
- operational problems
- unusual patterns
- accountability gaps
- inconsistencies between reports, emails, financial records,
  and operational data

Then produce high-value interview questions.

For every important interview question provide:

1. The question
2. Who should be asked
3. Evidence that motivated the question
4. Why the question matters

Questions should be specific, evidence-based, and difficult to
answer with a simple yes/no.

Then identify high-value AI use cases for the organization.

For every AI use case provide:

1. Use case
2. Problem solved
3. Data required
4. Expected benefit
5. Major risk or limitation

Do not invent facts that aren't present in the supplied material.

EXAM QUESTION:
""",

    "q4": """
You are helping with a short examination essay.

Analyze the exact exam question carefully.

First identify several ORIGINAL and less predictable angles that
could distinguish the answer from generic student responses.

Then select the strongest angle.

Write a polished final essay of NO MORE THAN 150 WORDS.

Requirements:

- directly answer the question
- have a clear original argument
- provide a concrete example where appropriate
- avoid generic AI phrases
- avoid unnecessary introduction
- avoid repetitive conclusion
- remain closely related to the assigned topic
- stay within 150 words

After the essay, provide the exact word count.

EXAM QUESTION:
"""
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    data = request.get_json()

    question_type = data.get("type")
    question = data.get("question", "").strip()
    material = data.get("material", "").strip()

    if not question:
        return jsonify({
            "error": "Please enter the exam question."
        }), 400

    if question_type not in PROMPTS:
        return jsonify({
            "error": "Invalid question type."
        }), 400

    prompt = (
        PROMPTS[question_type]
        + "\n\n"
        + question
        + "\n\nSUPPLIED MATERIAL:\n"
        + material
    )

    try:

        client = get_client()

        response = client.responses.create(
            model="openai/gpt-4.1-nano",
            input=prompt
        )

        return jsonify({
            "answer": response.output_text
        })

    except Exception as error:

        return jsonify({
            "error": str(error)
        }), 500


if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )
