// Parameter
const res_height = 28, res_width = 28;

// Fast Selector Function Define
var $ = function(id){return document.getElementById(id)};

// Global Objects
const canvas = $('demoCanvas');
var ctx = canvas.getContext("2d");

const canvas_size = canvas.parentElement.clientWidth - 20;

// Canvas Setup
const f_canvas = new fabric.Canvas('demoCanvas', {
  isDrawingMode: true,
  height: canvas_size,
  width: canvas_size,
  backgroundColor: 'white'
});

if(f_canvas.freeDrawingBrush){
  var brush = f_canvas.freeDrawingBrush;
  // brush.width = canvas_size / 28;
  brush.width = 10;
  brush.color = 'black';
}else{
  alert('您的瀏覽器不被支援');
  location.replace("/");
}

// Clear Button
$('btn-clean').addEventListener('click', () => {
    f_canvas.clear();
    f_canvas.backgroundColor = 'white';
});

$('submit-img').addEventListener('click', async () => {
    
    showLoadingScreen();


    // -- Get Limits --
    var lines = [];
    f_canvas.getObjects().forEach(function (obj) {
        if (obj.type === 'path') {
            lines.push(obj);
        }
    });

    console.log(lines);
    
    // Limit for one block => [left top right buttom] -> 遇到像是4之類需要兩個筆畫的物件，需要找上下左右的極值
    var limits = [canvas.width, canvas.height, 0, 0];

    lines.forEach((obj) => {
      // Left
      f_canvas.add(new fabric.Line([obj.left, 0, obj.left, canvas.height], {stroke: 'green'}))
      if(limits[0] > obj.left) limits[0] = obj.left;
      // Top
      f_canvas.add(new fabric.Line([0, obj.top, canvas.width, obj.top], {stroke: 'green'}))
      if(limits[1] > obj.top) limits[1] = obj.top;
      // Right
      f_canvas.add(new fabric.Line([obj.left + obj.width + obj.strokeWidth, 0, obj.left + obj.width + obj.strokeWidth, canvas.height], {stroke: 'green'}))
      if(limits[2] < obj.left + obj.width + obj.strokeWidth) limits[2] = obj.left + obj.width + obj.strokeWidth;
      // Button
      f_canvas.add(new fabric.Line([0, obj.top + obj.height + obj.strokeWidth, canvas.width, obj.top + obj.height + obj.strokeWidth], {stroke: 'green'}))
      if(limits[3] < obj.top + obj.height + obj.strokeWidth) limits[3] = obj.top + obj.height + obj.strokeWidth;
    })

    // Un-Comment to show Limits
    // f_canvas.add(new fabric.Line([limits[0], 0, limits[0], canvas.height], {stroke: 'red'}));
    // f_canvas.add(new fabric.Line([0, limits[1], canvas.width, limits[1]], {stroke: 'red'}));
    // f_canvas.add(new fabric.Line([limits[2], 0, limits[2], canvas.height], {stroke: 'red'}));
    // f_canvas.add(new fabric.Line([0, limits[3], canvas.width, limits[3]], {stroke: 'red'}));

    console.log(limits);

    // const size = ((limits[2] - limits[0]) > (limits[3] - limits[1]))? limits[2] - limits[0] : limits[3] - limits[1];
    // const image_input = await ctx.getImageData(limits[0], limits[1], limits[0] + size, limits[1] + size);
    const image_input = await ctx.getImageData(0, 0, canvas.width, canvas.height);

    // const canvas_data = await ctx.getImageData(limits[0], limits[1], limits[0] + size, limits[1] + size);
    const canvas_data = await resizeImageData(canvas, image_input, 28, 28, limits);
    // f_canvas.zoomToPoint({x: canvas.width / 2, y: canvas.height / 2}, 28 / size);
    // console.log(canvas_data.data);
    

    // == ColorFul Image Start ==
    
    // let r = new Array();
    // let g = new Array();
    // let b = new Array();

    // for (let i = 0; i < canvas_data.data.length; i += 4) {
    //   r.push(canvas_data.data[i+0]);
    //   g.push(canvas_data.data[i+1]);
    //   b.push(canvas_data.data[i+2]);
    // }

    // const formData = new FormData();
    // formData.append('color_r', r);
    // formData.append('color_g', g);
    // formData.append('color_b', b);
    
    // == ColorFul Imgae End ==


    // == GrayScale Image Start ==
    let gray = new Array();

    for (let i = 0; i < canvas_data.data.length; i += 4) {
      // GrayScale
      // gray.push(Math.round(0.299 * canvas_data.data[i] + 0.587 * canvas_data.data[i+1] + 0.114 *canvas_data.data[i+2])); 
      
      // Binary
      gray.push((0.299 * canvas_data.data[i] + 0.587 * canvas_data.data[i+1] + 0.114 *canvas_data.data[i+2]) > 50 ? 0 : 1); 
      // gray.push(' ');

    }

    console.log(gray);

    gray = await padding(gray, res_width, res_height);



    console.log(gray);

    const formData = new FormData();
    formData.append('gray_img', gray);
    formData.append('size_height', res_height);
    formData.append('size_width', res_width);

    // == GrayScale Image End ==


    // post form data
    const xhr = new XMLHttpRequest();

    // log response
    xhr.onload = () => {
        console.log(xhr.responseText);
    }

    // create and send the reqeust
    xhr.open('POST', '/apis/submit');
    // xhr.open('POST', 'https://495a4430-21de-4766-8cd9-c7a3c5d89cab.mock.pstmn.io');

    xhr.send(formData);

    xhr.onload = () => {
      if(xhr.status == 200){
        // document.documentElement.innerHTML = xhr.responseText;
        localStorage.setItem('prevData', xhr.responseText);
        hideLoadingScreen();
        window.location.href = '/result.html';

      }else{
        alert(`收到${xhr.status}回傳碼，原因：${xhr.responseText || '未知'}，請再試一次`);
        hideLoadingScreen();
      }
    }

    // xhr.onreadystatechange = () => { // 沒沙小路用
    //   console.log(xhr.readyState);
    // }

    xhr.onerror = () => {
      alert('發生問題，請再試一次');
      hideLoadingScreen();
    }

  }
);



function showLoadingScreen(){
  let popup = document.createElement('div');
  document.body.style.setProperty('overflow', 'hidden');
  document.body.appendChild(popup);
  popup.classList.add('animate__animated', 'animate__fadeIn');
  popup.style = `
    box-shadow: rgba(0, 0, 0, 0.2) 0px 3px 10px;
    background: rgba(0, 0, 0, 0.125);
    position: fixed;
    inset: 0px;
    backdrop-filter: blur(10px);
    `;
  popup.id = 'popup';
  popup.innerHTML = `
  <svg id="loading">
    <g>
      <path class="ld-l" fill="#39C0C4" d="M43.6,33.2h9.2V35H41.6V15.2h2V33.2z"/>
      <path class="ld-o" fill="#39C0C4" d="M74.7,25.1c0,1.5-0.3,2.9-0.8,4.2c-0.5,1.3-1.2,2.4-2.2,3.3c-0.9,0.9-2,1.6-3.3,2.2
        c-1.3,0.5-2.6,0.8-4.1,0.8s-2.8-0.3-4.1-0.8c-1.3-0.5-2.4-1.2-3.3-2.2s-1.6-2-2.2-3.3C54.3,28,54,26.6,54,25.1s0.3-2.9,0.8-4.2
        c0.5-1.3,1.2-2.4,2.2-3.3s2-1.6,3.3-2.2c1.3-0.5,2.6-0.8,4.1-0.8s2.8,0.3,4.1,0.8c1.3,0.5,2.4,1.2,3.3,2.2c0.9,0.9,1.6,2,2.2,3.3
        C74.4,22.2,74.7,23.6,74.7,25.1z M72.5,25.1c0-1.2-0.2-2.3-0.6-3.3c-0.4-1-0.9-2-1.6-2.8c-0.7-0.8-1.6-1.4-2.6-1.9
        c-1-0.5-2.2-0.7-3.4-0.7c-1.3,0-2.4,0.2-3.4,0.7c-1,0.5-1.9,1.1-2.6,1.9c-0.7,0.8-1.3,1.7-1.6,2.8c-0.4,1-0.6,2.1-0.6,3.3
        c0,1.2,0.2,2.3,0.6,3.3c0.4,1,0.9,2,1.6,2.7c0.7,0.8,1.6,1.4,2.6,1.9c1,0.5,2.2,0.7,3.4,0.7c1.3,0,2.4-0.2,3.4-0.7
        c1-0.5,1.9-1.1,2.6-1.9c0.7-0.8,1.3-1.7,1.6-2.7C72.4,27.4,72.5,26.3,72.5,25.1z"/>
      <path class="ld-a" fill="#39C0C4" d="M78.2,35H76l8.6-19.8h2L95.1,35h-2.2l-2.2-5.2H80.4L78.2,35z M81.1,27.9h8.7l-4.4-10.5L81.1,27.9z"/>
      <path class="ld-d" fill="#39C0C4" d="M98,15.2h6.6c1.2,0,2.5,0.2,3.7,0.6c1.2,0.4,2.4,1,3.4,1.9c1,0.8,1.8,1.9,2.4,3.1s0.9,2.7,0.9,4.3
        c0,1.7-0.3,3.1-0.9,4.3s-1.4,2.3-2.4,3.1c-1,0.8-2.1,1.5-3.4,1.9c-1.2,0.4-2.5,0.6-3.7,0.6H98V15.2z M100,33.2h4
        c1.5,0,2.8-0.2,3.9-0.7c1.1-0.5,2-1.1,2.8-1.8c0.7-0.8,1.3-1.6,1.6-2.6s0.5-2,0.5-3c0-1-0.2-2-0.5-3c-0.4-1-0.9-1.8-1.6-2.6
        c-0.7-0.8-1.6-1.4-2.8-1.8c-1.1-0.5-2.4-0.7-3.9-0.7h-4V33.2z"/>
      <path class="ld-i" fill="#39C0C4" d="M121.2,35h-2V15.2h2V35z"/>
      <path class="ld-n" fill="#39C0C4" d="M140.5,32.1L140.5,32.1l0.1-16.9h2V35h-2.5l-11.5-17.1h-0.1V35h-2V15.2h2.5L140.5,32.1z"/>
      <path class="ld-g" fill="#39C0C4" d="M162.9,18.8c-0.7-0.7-1.5-1.3-2.5-1.7c-1-0.4-2-0.6-3.3-0.6c-1.3,0-2.4,0.2-3.4,0.7s-1.9,1.1-2.6,1.9
        c-0.7,0.8-1.3,1.7-1.6,2.8c-0.4,1-0.6,2.1-0.6,3.3c0,1.2,0.2,2.3,0.6,3.3c0.4,1,0.9,2,1.6,2.7c0.7,0.8,1.6,1.4,2.6,1.9
        s2.2,0.7,3.4,0.7c1.1,0,2.1-0.1,3.1-0.4c0.9-0.2,1.7-0.5,2.3-0.9v-6h-4.6v-1.8h6.6v9c-1.1,0.7-2.2,1.1-3.5,1.5
        c-1.3,0.3-2.5,0.5-3.9,0.5c-1.5,0-2.9-0.3-4.1-0.8s-2.4-1.2-3.3-2.2c-0.9-0.9-1.6-2-2.1-3.3s-0.8-2.7-0.8-4.2s0.3-2.9,0.8-4.2
        c0.5-1.3,1.2-2.4,2.2-3.3c0.9-0.9,2-1.6,3.3-2.2c1.3-0.5,2.6-0.8,4.1-0.8c1.6,0,3,0.2,4.1,0.7s2.2,1.1,3,2L162.9,18.8z"/>
    </g>
  </svg>
  <svg width='182px' height='182px' xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid" class="uil-ripple"><rect x="0" y="0" width="100" height="100" fill="none" class="bk"></rect><g> <animate attributeName="opacity" dur="4s" repeatCount="indefinite" begin="0s" keyTimes="0;0.33;1" values="1;1;0"></animate><circle cx="50" cy="50" r="40" stroke="#eeeeee" fill="none" stroke-width="2" stroke-linecap="round"><animate attributeName="r" dur="4s" repeatCount="indefinite" begin="0s" keyTimes="0;0.33;1" values="0;22;44"></animate></circle></g><g><animate attributeName="opacity" dur="4s" repeatCount="indefinite" begin="2s" keyTimes="0;0.33;1" values="1;1;0"></animate><circle cx="50" cy="50" r="40" stroke="#eeeeee" fill="none" stroke-width="2" stroke-linecap="round"><animate attributeName="r" dur="4s" repeatCount="indefinite" begin="2s" keyTimes="0;0.33;1" values="0;22;44"></animate></circle></g></svg>`;
  
}


function hideLoadingScreen(){
  $('popup').remove();
  document.body.style.setProperty('overflow', 'unset');
}

// async function resizeImageData (f_canvas, imageData, width, height) {
//   const resizeWidth = width >> 0
//   const resizeHeight = height >> 0
//   const ibm = await window.createImageBitmap(imageData, 0, 0, imageData.width, imageData.height, {
//     resizeWidth, resizeHeight
//   })
//   const _canvas = document.createElement('canvas')
//   _canvas.width = resizeWidth
//   _canvas.height = resizeHeight
//   const _ctx = _canvas.getContext('2d')
//   _ctx.drawImage(ibm, 0, 0)
//   return _ctx.getImageData(0, 0, resizeWidth, resizeHeight)
// }

async function resizeImageData (f_canvas, imageData, width, height, limits) {
  const resizeWidth = width >> 0;
  const resizeHeight = height >> 0;
  const _canvas = document.createElement('canvas');
  _canvas.width = resizeWidth;
  _canvas.height = resizeHeight;
  const _ctx = _canvas.getContext('2d');
  _ctx.fillStyle = 'white';
  _ctx.fillRect(0, 0, resizeWidth, resizeHeight);
  _ctx.drawImage(f_canvas, limits[0], limits[1], limits[2], limits[3], 1, 1, resizeWidth - 1, resizeHeight - 1);
  return _ctx.getImageData(0, 0, resizeWidth, resizeHeight);
}


async function padding(imageArray, width, height) {
  let ret = '';

  // Assuming 'imageArray' only contains 0 and 1

  for(let i = 0; i < width + 2; i += 1){
    ret += 1 - i % 2;
    ret += ' '
  }
  
  ret = ret.substring(0, ret.length - 1);
  ret += '\n';

  // ret += ' ';
  for(let i = 0; i < width; i += 1){
    ret += i % 2;

    ret += ' ';

    for(let j = 0; j < height; j += 1){
      ret += imageArray[i * width + j];

      ret += ' ';
    }
    ret += 1 - i % 2;
    ret += '\n';
    // ret += ' ';
  }
  for(let i = 0; i < width + 2; i += 1){
    ret += i % 2;
    ret += ' ';
  }
  ret = ret.substring(0, ret.length - 1);
  return ret;
}