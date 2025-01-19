// Select the form element and the result display area by their IDs
const form = document.getElementById("predictionForm");
const resultDiv = document.getElementById("result");

// Listen for the form's 'submit' event
form.addEventListener("submit", async (event) => {
    // Prevent the default form submission (which would reload the page)
    event.preventDefault();

    // Gather all fields from the form into a FormData object
    const form = event.target;
    const formData = new FormData(form);

    // Construct a plain JavaScript object from the form data for JSON serialization
    const payload = {};
    for (const [key, value] of formData.entries()) {
        payload[key] = value;
    }

    try {
        // Send the JSON payload to the '/predict' endpoint via a POST request
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        // Convert the response into a JavaScript object
        const data = await response.json();

        // Display the prediction result in the page
        resultDiv.innerHTML = `<p><strong>Predicted Price:</strong> ${data.predicted_price}</p>`;
    } catch (err) {
        // If there's any error, log it and show a user-friendly message
        console.error("Prediction error:", err);
        resultDiv.innerHTML = `<p style="color: red;">Something went wrong. Check console for details.</p>`;
    }
});
