document.addEventListener('DOMContentLoaded', () => {
    // Tab Switching Logic
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            // Remove active class from all
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            // Add active class to clicked
            btn.classList.add('active');
            const targetId = btn.getAttribute('data-target');
            document.getElementById(targetId).classList.add('active');
        });
    });

    const analyzeBtn = document.getElementById('analyzeBtn');
    const reviewInput = document.getElementById('reviewInput');
    const resultContainer = document.getElementById('resultContainer');
    const sentimentBadge = document.getElementById('sentimentBadge');
    const sentimentText = document.getElementById('sentimentText');
    const cleanTextDisplay = document.getElementById('cleanTextDisplay');
    const svgIcon = document.querySelector('.sentiment-icon');

    const icons = {
        'Positive': '<path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"></path>',
        'Negative': '<path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"></path>',
        'Neutral': '<circle cx="12" cy="12" r="10"></circle><line x1="8" y1="12" x2="16" y2="12"></line>'
    };

    analyzeBtn.addEventListener('click', async () => {
        const text = reviewInput.value.trim();
        
        if (!text) {
            // Briefly shake the input if empty
            reviewInput.style.transform = 'translateX(-5px)';
            setTimeout(() => reviewInput.style.transform = 'translateX(5px)', 50);
            setTimeout(() => reviewInput.style.transform = 'translateX(0)', 100);
            return;
        }

        // Show loading state
        analyzeBtn.classList.add('loading');
        analyzeBtn.disabled = true;
        resultContainer.classList.add('hidden');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text })
            });

            const data = await response.json();

            if (response.ok) {
                // Update UI with results
                const sentiment = data.sentiment;
                
                sentimentText.textContent = sentiment;
                cleanTextDisplay.textContent = data.clean_text || "N/A";
                
                // Update Badge Class
                sentimentBadge.className = 'sentiment-badge'; // reset
                sentimentBadge.classList.add(`sentiment-${sentiment}`);
                
                // Update Icon
                svgIcon.innerHTML = icons[sentiment] || icons['Neutral'];

                // Show result container
                resultContainer.classList.remove('hidden');
                
                // Scroll to result slightly if needed
                setTimeout(() => {
                    resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }, 100);
            } else {
                alert(`Error: ${data.error || 'Failed to analyze text'}`);
            }

        } catch (error) {
            console.error('Error:', error);
            alert('An unexpected error occurred while communicating with the server.');
        } finally {
            // Remove loading state
            analyzeBtn.classList.remove('loading');
            analyzeBtn.disabled = false;
        }
    });

    // Allow Enter key to submit (Ctrl+Enter for textareas is good UX)
    reviewInput.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            analyzeBtn.click();
        }
    });
});
