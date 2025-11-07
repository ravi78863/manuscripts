document.addEventListener('DOMContentLoaded', function () {
    const uploadForm = document.getElementById('uploadForm');
    const manuscriptImageInput = document.getElementById('manuscriptImage');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const errorMessage = document.getElementById('errorMessage');
    const resultsDisplay = document.getElementById('resultsDisplay');
    const originalImage = document.getElementById('originalImage');
    const heatmapImage = document.getElementById('heatmapImage');
    const overlayImage = document.getElementById('overlayImage');
    const damagePercentage = document.getElementById('damagePercentage');

    uploadForm.addEventListener('submit', async function (event) {
        event.preventDefault(); // Prevent default form submission

        const file = manuscriptImageInput.files[0];
        if (!file) {
            alert('Please select an image file.');
            return;
        }

        // Show loading spinner, hide previous results/errors
        loadingSpinner.classList.remove('d-none');
        resultsDisplay.classList.add('d-none');
        errorMessage.classList.add('d-none');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
            }

            const data = await response.json();

            // Display results
            originalImage.src = `data:image/jpeg;base64,${data.original_image}`;
            heatmapImage.src = `data:image/jpeg;base64,${data.heatmap_image}`;
            overlayImage.src = `data:image/jpeg;base64,${data.overlay_image}`;
            damagePercentage.textContent = `Predicted Physical Damage: ${data.damage_percentage}`;

            resultsDisplay.classList.remove('d-none'); // Show results section

        } catch (error) {
            console.error('Upload failed:', error);
            errorMessage.textContent = `Error: ${error.message}`;
            errorMessage.classList.remove('d-none');
        } finally {
            loadingSpinner.classList.add('d-none'); // Hide loading spinner
        }
    });
});