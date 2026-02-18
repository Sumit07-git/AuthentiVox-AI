// Main JavaScript - DeepGuard AI
// Handle navigation, dark mode, and global interactions

document.addEventListener('DOMContentLoaded', function() {
    // Dark Mode Toggle
    initDarkMode();
    
    // Mobile Navigation Toggle
    const navToggle = document.querySelector('.nav-toggle');
    const navMenu = document.querySelector('.nav-menu');
    
    if (navToggle) {
        navToggle.addEventListener('click', function() {
            navMenu.classList.toggle('active');
        });
    }
    
    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Add animation on scroll
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -100px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);
    
    // Observe elements with fade-in class
    document.querySelectorAll('.feature-card, .info-point').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
});

// Dark Mode Functions
function initDarkMode() {
    // Check for saved theme preference or default to light mode
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    
    // Create dark mode toggle button
    createDarkModeToggle();
    
    // Update toggle button state
    updateToggleButton(savedTheme === 'dark');
}

function createDarkModeToggle() {
    // Check if toggle already exists
    if (document.getElementById('darkModeToggle')) return;
    
    // Create toggle button
    const toggleBtn = document.createElement('button');
    toggleBtn.id = 'darkModeToggle';
    toggleBtn.className = 'dark-mode-toggle';
    toggleBtn.setAttribute('aria-label', 'Toggle dark mode');
    toggleBtn.innerHTML = '<i class="fas fa-moon"></i>';
    
    // Add to nav menu
    const navMenu = document.querySelector('.nav-menu');
    if (navMenu) {
        const li = document.createElement('li');
        li.appendChild(toggleBtn);
        navMenu.appendChild(li);
    }
    
    // Add click handler
    toggleBtn.addEventListener('click', function(e) {
        e.preventDefault();
        toggleDarkMode();
    });
}

function toggleDarkMode() {
    const html = document.documentElement;
    const currentTheme = html.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    html.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    
    updateToggleButton(newTheme === 'dark');
    
    // Add transition animation
    html.style.transition = 'background-color 0.3s ease, color 0.3s ease';
}

function updateToggleButton(isDark) {
    const toggleBtn = document.getElementById('darkModeToggle');
    if (toggleBtn) {
        const icon = toggleBtn.querySelector('i');
        if (isDark) {
            icon.className = 'fas fa-sun';
            toggleBtn.setAttribute('aria-label', 'Switch to light mode');
        } else {
            icon.className = 'fas fa-moon';
            toggleBtn.setAttribute('aria-label', 'Switch to dark mode');
        }
    }
}