const video = document.getElementById("video");
const resultDiv = document.getElementById("result");

navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
    video.srcObject = stream;
});

setInterval(async () => {
    if (video.readyState !== 4) return;

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);

    const blob = await new Promise(resolve =>
        canvas.toBlob(resolve, "image/jpeg")
    );

    const formData = new FormData();
    formData.append("image", blob, "frame.jpg");

    const response = await fetch("/predict", {
        method: "POST",
        body: formData
    });

    const data = await response.json();
    resultDiv.textContent = "Gesture: " + data.gesture;
}, 200);
