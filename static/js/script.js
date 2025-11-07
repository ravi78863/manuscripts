document.addEventListener('DOMContentLoaded', function () {

    // --- Get all elements ---
    const uploadForm = document.getElementById('uploadForm');
    const manuscriptImageInput = document.getElementById('manuscriptImage');
    const dropZone = document.getElementById('dropZone');

    const loadingSpinner = document.getElementById('loadingSpinner');
    const errorMessage = document.getElementById('errorMessage');

    const resultsDisplay = document.getElementById('resultsDisplay');
    const originalImage = document.getElementById('originalImage');
    const heatmapImage = document.getElementById('heatmapImage');
    const overlayImage = document.getElementById('overlayImage');
    const damagePercentage = document.getElementById('damagePercentage');

    // --- Event Listeners ---

    // 1. Click on the drop zone to open file dialog
    dropZone.addEventListener('click', () => {
        manuscriptImageInput.click();
    });

    // 2. When a file is chosen via the file dialog
    manuscriptImageInput.addEventListener('change', () => {
        const file = manuscriptImageInput.files[0];
        if (file) {
            handleUpload(file);
        }
    });

    // 3. Drag-and-drop: When dragging over the zone
    dropZone.addEventListener('dragover', (event) => {
        event.preventDefault(); // Prevent default browser behavior
        dropZone.classList.add('drop-zone-over');
    });

    // 4. Drag-and-drop: When leaving the zone
    dropZone.addEventListener('dragleave', (event) => {
        event.preventDefault();
        dropZone.classList.remove('drop-zone-over');
    });

    // 5. Drag-and-drop: When dropping the file
    dropZone.addEventListener('drop', (event) => {
        event.preventDefault();
        dropZone.classList.remove('drop-zone-over');

        const file = event.dataTransfer.files[0];
        if (file) {
            manuscriptImageInput.files = event.dataTransfer.files; // Sync the file list
            handleUpload(file);
        }
    });

    // --- Main Upload Function ---
    // This function is called by both "drop" and "file selection"

    async function handleUpload(file) {
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
            // Scroll to results on mobile
            if (window.innerWidth < 768) {
                resultsDisplay.scrollIntoView({ behavior: 'smooth' });
            }


        } catch (error) {
            console.error('Upload failed:', error);
            errorMessage.textContent = `Error: ${error.message}`;
            errorMessage.classList.remove('d-none');
        } finally {
            loadingSpinner.classList.add('d-none'); // Hide loading spinner
            uploadForm.reset(); // Clear the file input
        }
    }
});