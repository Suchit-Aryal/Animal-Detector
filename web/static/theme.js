// Set the theme before the page paints, so there is no flash. Saved choice wins over the system setting.
// A separate file (not inline) so the hosted site can use a strict Content-Security-Policy.
try { var t = localStorage.getItem("theme"); if (t === "light" || t === "dark") document.documentElement.dataset.theme = t; } catch (e) {}
