document.getElementById('btn-clean').addEventListener('click', (e) => {
    myCanvas.clear()
});

document.getElementById('submit-img').addEventListener('click', (e) => {

    var canvas = document.getElementById("demoCanvas");
    var canvas_url = canvas.toDataURL();
    console.log(canvas_url);
    
    $.ajax({
        type: "POST",
        url: "/submit",
        data:{
          imageBase64: dataURL
        }
      }).done(function() {
        console.log('sent');
      });

    // myCanvas.clear()
    // document.body.style.display = "block";
});

