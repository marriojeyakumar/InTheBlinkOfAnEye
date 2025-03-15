const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const snapButton = document.getElementById('snap');
const cameraSound = document.getElementById('cameraSound');
const context = canvas.getContext('2d');

if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
  navigator.mediaDevices.getUserMedia({ video: true })
    .then(function(stream) {
      if ("srcObject" in video) {
        video.srcObject = stream;
      } else {
        video.src = window.URL.createObjectURL(stream);
      }
      video.play();
    })
    .catch(function(error) {
      console.error("Error accessing the camera", error);
      alert("Unable to access the camera. Please check your settings.");
    });
} else {
  alert("getUserMedia is not supported by your browser.");
}

snapButton.addEventListener("click", function() {
  cameraSound.play();
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
});
