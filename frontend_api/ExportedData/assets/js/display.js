var $ = function(id){return document.getElementById(id)};
var g = function(data){
    let ret = JSON.parse(localStorage.getItem(data));
    if(ret == null){
        window.location.href = '/demo.html'; // 回到上一頁
    }
    return ret;
};

var prev_result = g('prevData');

$('show-result').innerHTML = prev_result.result;
$('time-col1').innerHTML = Math.round(prev_result.total_time * 10e5) / 10e2 + 'ms';
$('time-col2').innerHTML = Math.round(prev_result.data_prep_time * 10e5) / 10e2 + 'ms';
$('time-col3').innerHTML = Math.round(prev_result.conv_time * 10e5) / 10e2 + 'ms';
$('time-col4').innerHTML = Math.round(prev_result.pool_time * 10e5) / 10e2 + 'ms';

window.addEventListener('load', (e) => {
    var prob = prev_result.probability_arr;
    chart = document.querySelector('canvas').chart;
    for(let i = 0; i < prob.length; i = i + 1){
        chart.data.datasets[0].data.push(prob[i]);
        chart.data.labels.push(i);
    }

    prob.sort(function(a, b){return a-b});
    if((prob[prob.length - 1] / prob[prob.length - 2]) > 10){
        chart.options.scales.yAxes[0].type = 'logarithmic';
        chart.options.scales.yAxes[0].display = false;
        $('info-text').innerHTML += '<br><strong>由於資料差距過大，此表以對數刻度表示y軸。</strong>';
    }else{
        $('info-text').innerHTML += '<br>前兩筆資料差距不大，未使用對數刻度表示y軸';
    }
    chart.update();
});

