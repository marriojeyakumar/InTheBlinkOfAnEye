// script.js

function refreshCapturedImage() {
  const capturedImg = document.getElementById("captured");
  // Append a timestamp query string to bypass browser cache.
  capturedImg.src = "/static/captured.jpg?t=" + new Date().getTime();
}

// Refresh the captured image every second.
setInterval(refreshCapturedImage, 1000);
