function refreshCapturedImage() {
  const capturedImg = document.getElementById("captured");
  capturedImg.src = "/static/captured.jpg?t=" + new Date().getTime();
}

setInterval(refreshCapturedImage, 1000);
