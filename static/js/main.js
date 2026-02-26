


document.addEventListener('DOMContentLoaded', function() {
    
    initDarkMode();
    
    
    const navToggle = document.querySelector('.nav-toggle');
    const navMenu = document.querySelector('.nav-menu');
    const navbar = document.querySelector('.navbar');
    
    if (navToggle && navMenu) {
        
        navToggle.addEventListener('click', function(e) {
            e.stopPropagation();
            navMenu.classList.toggle('active');
        });
        
        
        document.addEventListener('click', function(e) {
            if (!navbar.contains(e.target)) {
                navMenu.classList.remove('active');
            }
        });
        
        
        navMenu.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', function() {
                navMenu.classList.remove('active');
            });
        });
    }
    
    
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
    
    
    document.querySelectorAll('.feature-card, .info-point').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(20px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
});


function initDarkMode() {
    
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    
    
    createDarkModeToggle();
    
    
    updateToggleButton(savedTheme === 'dark');
}

function createDarkModeToggle() {
    
    if (document.getElementById('darkModeToggle')) return;
    
    
    const toggleBtn = document.createElement('button');
    toggleBtn.id = 'darkModeToggle';
    toggleBtn.className = 'dark-mode-toggle';
    toggleBtn.setAttribute('aria-label', 'Toggle dark mode');
    toggleBtn.innerHTML = '<i class="fas fa-moon"></i>';
    
    
    const navMenu = document.querySelector('.nav-menu');
    if (navMenu) {
        const li = document.createElement('li');
        li.appendChild(toggleBtn);
        navMenu.appendChild(li);
    }
    
    
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

const track = document.getElementById("sliderTrack");
const pauseBtn = document.getElementById("pauseBtn");

let position = 0;
const totalCards = track.children.length;


function getCardWidth() {
    const card = document.querySelector(".template-card");
    const cardStyle = window.getComputedStyle(card);
    const trackStyle = window.getComputedStyle(track);

    const gap = parseInt(trackStyle.gap) || 0;
    return card.offsetWidth + gap;
}


function moveSlide(direction) {
    const cardWidth = getCardWidth();

    position += direction;

    
    const wrapperWidth = document.querySelector(".slider-wrapper").offsetWidth;
    const visibleCards = Math.floor(wrapperWidth / cardWidth);

    if (position < 0) position = totalCards - visibleCards;
    if (position > totalCards - visibleCards) position = 0;

    track.style.transform = `translateX(-${position * cardWidth}px)`;
}


let autoSlide = setInterval(() => moveSlide(1), 2500);
let isPaused = false;


function toggleAuto() {
    if (isPaused) {
        autoSlide = setInterval(() => moveSlide(1), 2500);
        pauseBtn.innerHTML = "⏸";
    } else {
        clearInterval(autoSlide);
        pauseBtn.innerHTML = "▶";
    }
    isPaused = !isPaused;
}

