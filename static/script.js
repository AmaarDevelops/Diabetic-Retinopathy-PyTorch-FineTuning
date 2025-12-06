// /static/script.js
document.getElementById('upload-form').addEventListener('submit', async function(event) {
    event.preventDefault(); // Stop the default form submission

    const form = event.target;
    const formData = new FormData(form);
    const resultBox = document.getElementById('result-box');
    const predictionText = document.getElementById('prediction-text');
    const submitBtn = document.getElementById('submit-btn');

    // 1. UI Feedback (Disable button, show loading)
    submitBtn.disabled = true;
    submitBtn.textContent = 'Diagnosing...';
    predictionText.textContent = 'Processing image and running model...';
    resultBox.style.borderColor = '#ffc107'; // Yellow border for loading

    try {
        // 2. Send data to Flask backend
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        // 3. Handle response
        if (data.success) {
            predictionText.innerHTML = `**Prediction:** ${data.prediction}`;
            resultBox.style.borderColor = '#28a745'; // Green border for success
        } else {
            predictionText.textContent = `Error: ${data.error}`;
            resultBox.style.borderColor = '#dc3545'; // Red border for error
        }

    } catch (error) {
        console.error('Fetch error:', error);
        predictionText.textContent = 'A network or server error occurred.';
        resultBox.style.borderColor = '#dc3545';
    } finally {
        // 4. Reset UI
        submitBtn.disabled = false;
        submitBtn.textContent = 'Diagnose Image';
    }
});
