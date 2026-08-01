async function analyze() {

    const button =
        document.getElementById("analyzeButton");

    const questionType =
        document.getElementById("questionType").value;

    const question =
        document.getElementById("question").value;

    const material =
        document.getElementById("material").value;

    const result =
        document.getElementById("result");


    if (!question.trim()) {

        result.textContent =
            "Please paste the exam question.";

        return;
    }


    button.disabled = true;

    button.textContent = "Analyzing...";

    result.textContent =
        "Generating response. Please wait...";


    try {

        const response = await fetch(
            "/analyze",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    type: questionType,
                    question: question,
                    material: material
                })
            }
        );


        const data = await response.json();


        if (data.error) {

            result.textContent =
                "ERROR:\n\n" + data.error;

        } else {

            result.textContent =
                data.answer;

        }


    } catch (error) {

        result.textContent =
            "ERROR:\n\n" + error.message;

    }


    button.disabled = false;

    button.textContent =
        "Generate Analysis";
}


function copyResult() {

    const result =
        document.getElementById("result").textContent;

    navigator.clipboard.writeText(result);

}
