// Tab toggle logic
document.addEventListener("DOMContentLoaded", function() {
    const loginTab = document.getElementById('login-tab');
    const registerTab = document.getElementById('register-tab');
    const loginForm = document.getElementById('login-form');
    const registerForm = document.getElementById('register-form');
    if (loginTab && registerTab && loginForm && registerForm) {
        loginTab.onclick = function() {
            this.classList.add('active');
            registerTab.classList.remove('active');
            loginForm.style.display = '';
            registerForm.style.display = 'none';
        };
        registerTab.onclick = function() {
            this.classList.add('active');
            loginTab.classList.remove('active');
            loginForm.style.display = 'none';
            registerForm.style.display = '';
        };
    }

    // Password strength indicator
    const passwordInput = document.getElementById('register-password');
    const strengthDiv = document.getElementById('password-strength');
    if (passwordInput && strengthDiv) {
        passwordInput.addEventListener('input', function() {
            const val = passwordInput.value;
            let strength = 0;
            if (val.length >= 8) strength++;
            if (/[A-Z]/.test(val)) strength++;
            if (/[0-9]/.test(val)) strength++;
            if (/[^A-Za-z0-9]/.test(val)) strength++;
            let msg = '';
            let color = '';
            switch (strength) {
                case 0:
                case 1:
                    msg = 'Weak';
                    color = '#EF5350';
                    break;
                case 2:
                    msg = 'Fair';
                    color = '#FFA726';
                    break;
                case 3:
                    msg = 'Good';
                    color = '#2196F3';
                    break;
                case 4:
                    msg = 'Strong';
                    color = '#66BB6A';
                    break;
            }
            strengthDiv.textContent = msg;
            strengthDiv.style.color = color;
            strengthDiv.style.fontWeight = 'bold';
            strengthDiv.style.marginTop = '0.3em';
        });
    }
});
