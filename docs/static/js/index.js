window.HELP_IMPROVE_VIDEOJS = false;

// More Works Dropdown Functionality
function toggleMoreWorks() {
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');

    if (dropdown.classList.contains('show')) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    } else {
        dropdown.classList.add('show');
        button.classList.add('active');
    }
}

// Close dropdown when clicking outside
document.addEventListener('click', function(event) {
    const container = document.querySelector('.more-works-container');
    const dropdown = document.getElementById('moreWorksDropdown');
    const button = document.querySelector('.more-works-btn');

    if (container && !container.contains(event.target)) {
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Close dropdown on escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const dropdown = document.getElementById('moreWorksDropdown');
        const button = document.querySelector('.more-works-btn');
        dropdown.classList.remove('show');
        button.classList.remove('active');
    }
});

// Citation modal — open / close
function openCiteModal() {
    const modal = document.getElementById('cite-modal');
    if (modal) {
        modal.hidden = false;
        document.body.style.overflow = 'hidden';
    }
}

function closeCiteModal() {
    const modal = document.getElementById('cite-modal');
    if (modal) {
        modal.hidden = true;
        document.body.style.overflow = '';
    }
}

// Close modal on Escape
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const modal = document.getElementById('cite-modal');
        if (modal && !modal.hidden) closeCiteModal();
    }
});

// Copy BibTeX to clipboard
function copyBibTeX() {
    const bibtexElement = document.getElementById('bibtex-code');
    const button = document.querySelector('.copy-bibtex-btn');
    const copyText = button.querySelector('.copy-text');

    if (bibtexElement) {
        navigator.clipboard.writeText(bibtexElement.textContent).then(function() {
            // Success feedback
            button.classList.add('copied');
            copyText.textContent = 'Copied!';

            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        }).catch(function(err) {
            console.error('Failed to copy: ', err);
            // Fallback for older browsers
            const textArea = document.createElement('textarea');
            textArea.value = bibtexElement.textContent;
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);

            button.classList.add('copied');
            copyText.textContent = 'Copied!';
            setTimeout(function() {
                button.classList.remove('copied');
                copyText.textContent = 'Copy';
            }, 2000);
        });
    }
}

// Video carousel autoplay when in view
function setupVideoCarouselAutoplay() {
    const carouselVideos = document.querySelectorAll('.results-carousel video');

    if (carouselVideos.length === 0) return;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            const video = entry.target;
            if (entry.isIntersecting) {
                // Video is in view, play it
                video.play().catch(e => {
                    // Autoplay failed, probably due to browser policy
                    console.log('Autoplay prevented:', e);
                });
            } else {
                // Video is out of view, pause it
                video.pause();
            }
        });
    }, {
        threshold: 0.5 // Trigger when 50% of the video is visible
    });

    carouselVideos.forEach(video => {
        observer.observe(video);
    });
}

// Model browser — switch active tab + pane
function setupModelBrowser() {
    document.querySelectorAll('.model-browser').forEach(browser => {
        const tabs = browser.querySelectorAll('.model-tab');
        const panes = browser.querySelectorAll('.model-pane');
        if (!tabs.length || !panes.length) return;

        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                const target = tab.dataset.model;
                tabs.forEach(t => {
                    const active = t === tab;
                    t.classList.toggle('is-active', active);
                    t.setAttribute('aria-selected', active ? 'true' : 'false');
                });
                panes.forEach(p => {
                    p.classList.toggle('is-active', p.dataset.model === target);
                });
            });
        });
    });
}

// Section TOC nav — active state on scroll + smooth jump (desktop sidebar + mobile drawer)
function setupTocNav() {
    const links = document.querySelectorAll('.toc-drawer-list a[data-section]');
    if (!links.length) return;

    const sectionToLinks = new Map();
    links.forEach(link => {
        const sec = document.getElementById(link.dataset.section);
        if (!sec) return;
        if (!sectionToLinks.has(sec)) sectionToLinks.set(sec, []);
        sectionToLinks.get(sec).push(link);
    });

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (!entry.isIntersecting) return;
            const targets = sectionToLinks.get(entry.target);
            if (!targets) return;
            links.forEach(l => l.classList.remove('active'));
            targets.forEach(l => l.classList.add('active'));
        });
    }, {
        rootMargin: '-25% 0px -65% 0px',
        threshold: 0
    });

    sectionToLinks.forEach((_, sec) => observer.observe(sec));

    links.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const target = document.getElementById(link.dataset.section);
            if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            // Close mobile drawer if the click originated from it
            if (link.closest('.toc-drawer-list')) closeTocDrawer();
        });
    });
}

// Mobile TOC drawer open/close
function openTocDrawer() {
    const drawer = document.getElementById('toc-drawer');
    if (drawer) {
        drawer.hidden = false;
        document.body.style.overflow = 'hidden';
    }
}

function closeTocDrawer() {
    const drawer = document.getElementById('toc-drawer');
    if (drawer) {
        drawer.hidden = true;
        document.body.style.overflow = '';
    }
}

// Close drawer on Escape
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        const drawer = document.getElementById('toc-drawer');
        if (drawer && !drawer.hidden) closeTocDrawer();
    }
});

$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
		slidesToScroll: 1,
		slidesToShow: 1,
		loop: true,
		infinite: true,
		autoplay: true,
		autoplaySpeed: 5000,
    }

	// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    bulmaSlider.attach();

    // Setup video autoplay for carousel
    setupVideoCarouselAutoplay();

    // Setup section TOC nav
    setupTocNav();

    // Setup model browser tabs
    setupModelBrowser();

})
