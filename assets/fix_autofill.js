console.log("📢 fix_autofill.js loaded (Edge-compatible)");

window.addEventListener('DOMContentLoaded', () => {
    // Delay to let autofill happen
    setTimeout(() => {
        const username = document.querySelector('#username');
        const password = document.querySelector('#password');
        const loginBtn = document.querySelector('#login-button');

        if (username && password && loginBtn) {
            loginBtn.addEventListener('click', function () {
                console.log("🔐 Forcing input/change events on autofilled fields (Edge fix)");

                // Manually trigger input and change events so Dash sees the values
                const forceEvent = (el) => {
                    ['input', 'change'].forEach(eventType => {
                        el.dispatchEvent(new Event(eventType, { bubbles: true }));
                    });
                };

                // Set .value to itself to trigger dirty checking
                username.value = username.value;
                password.value = password.value;

                // Force the events
                forceEvent(username);
                forceEvent(password);
            });
        }
    }, 500); // small delay to wait for autofill to complete
});
