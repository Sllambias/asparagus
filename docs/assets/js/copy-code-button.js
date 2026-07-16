document.addEventListener('DOMContentLoaded', function() {
    // SVG Icons
    const copyIconSVG = `<svg class="copy-icon" fill="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z"/></svg>`;
    const checkIconSVG = `<svg class="check-icon" fill="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z"/></svg>`;

    // Find all code blocks
    const codeBlocks = document.querySelectorAll('pre');

    codeBlocks.forEach((pre) => {
        // Create wrapper
        const wrapper = document.createElement('div');
        wrapper.className = 'code-block-wrapper';
        
        // Insert wrapper before pre
        pre.parentNode.insertBefore(wrapper, pre);
        
        // Move pre into wrapper
        wrapper.appendChild(pre);

        // Create copy button
        const button = document.createElement('button');
        button.className = 'copy-button';
        button.type = 'button';
        button.innerHTML = copyIconSVG + '<span class="copy-button-text">Copy</span>';
        button.title = 'Copy code to clipboard';

        // Add button to wrapper
        wrapper.appendChild(button);

        // Copy functionality
        button.addEventListener('click', async function(e) {
            e.preventDefault();
            
            // Get code text from the pre element
            const code = pre.innerText;
            
            try {
                // Copy to clipboard
                await navigator.clipboard.writeText(code);
                
                // Change button appearance
                button.classList.add('copied');
                button.innerHTML = checkIconSVG + '<span class="copy-button-text">Copied!</span>';
                
                // Reset button after 2 seconds
                setTimeout(() => {
                    button.classList.remove('copied');
                    button.innerHTML = copyIconSVG + '<span class="copy-button-text">Copy</span>';
                }, 2000);
            } catch (err) {
                console.error('Failed to copy code:', err);
                button.innerHTML = '❌ <span class="copy-button-text">Failed</span>';
                setTimeout(() => {
                    button.innerHTML = copyIconSVG + '<span class="copy-button-text">Copy</span>';
                }, 2000);
            }
        });
    });

    // Watch for theme changes (for mkdocs-shadcn theme toggle)
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.attributeName === 'class' || mutation.attributeName === 'data-theme') {
                // Force CSS variables update by triggering a repaint
                document.documentElement.style.colorScheme = document.documentElement.classList.contains('dark') ? 'dark' : 'light';
            }
        });
    });

    // Observe html element for class/attribute changes
    observer.observe(document.documentElement, {
        attributes: true,
        attributeFilter: ['class', 'data-theme']
    });
});