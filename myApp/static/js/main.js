document.addEventListener('DOMContentLoaded', function() {
    const searchForm = document.querySelector('.search-form');
    const searchBtn = document.querySelector('.search-btn');
    const loadingSpinner = document.querySelector('.loading-spinner');

    if (searchForm) {
        searchForm.addEventListener('submit', function(e) {
            // Show loading state
            searchBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Searching...';
            searchBtn.disabled = true;

            // Show loading spinner
            if (loadingSpinner) {
                loadingSpinner.style.display = 'block';
            }
        });
    }

    // Product card animations
    const productCards = document.querySelectorAll('.product-card');
    productCards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';

        setTimeout(() => {
            card.style.transition = 'all 0.5s ease';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, index * 100);
    });
});

// Search suggestions (optional enhancement)
function initializeSearchSuggestions() {
    const searchInput = document.querySelector('.search-input');
    if (searchInput) {
        searchInput.addEventListener('input', function(e) {
            const query = e.target.value;
            if (query.length > 2) {
                // Add autocomplete functionality here
                console.log('Searching for:', query);
            }
        });
    }
}