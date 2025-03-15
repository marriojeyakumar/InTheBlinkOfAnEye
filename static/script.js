let lastTimestamp = 0;
const capturedImg = document.getElementById("captured");
const shutterSound = document.getElementById("shutterSound");

function refreshCapturedImage() {
  capturedImg.src = "/static/captured.jpg?t=" + new Date().getTime();
}

function checkPhotoTimestamp() {
  fetch("/photo_timestamp")
    .then((response) => response.json())
    .then((data) => {
      if (data.timestamp > lastTimestamp) {
        lastTimestamp = data.timestamp;
        shutterSound.play();
      }
    })
    .catch((error) => console.error("Error fetching timestamp:", error));
}

setInterval(refreshCapturedImage, 1000);
setInterval(checkPhotoTimestamp, 1000);
