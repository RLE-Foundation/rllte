# Hsuanwu - Overview
<div align=center>
<img src='./assets/images/logo.png' style="width: 90%">
</div>

<img src="https://img.shields.io/badge/License-Apache-blue">
<img src="https://img.shields.io/badge/Python->=3.8-brightgreen"> <img src="https://img.shields.io/badge/DMC Suite-1.0.5-blue">
<img src="https://img.shields.io/badge/Docs-Developing-%23ff595e"> 

**Hsuanwu: Long-Term Evolution Project of Reinforcement Learning** is inspired by the long-term evolution (LTE) standard project in telecommunications, which aims to track the latest research progress in reinforcement learning (RL) and provide stable baselines.
The highlight features of Hsuanwu:

- 🧱 Complete decoupling of RL algorithms, and each method can be invoked separately;
- 📚 Large number of reusable bechmarking implementations ([See Benchmarks](benchmarks));
- 🛠️ Support for RL model engineering deployment (C++ API);

<div align=center>
<script type="text/javascript" src="https://code.jquery.com/jquery-3.6.3.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/echarts@5.4.1/dist/echarts.min.js"></script>
<div id="main" style="width: 750px; height:400px;">
    <script type="text/javascript">
var ROOT_PATH = 'https://echarts.apache.org/examples';    
var chartDom = document.getElementById('main');
var myChart = echarts.init(chartDom);
var option;
myChart.showLoading();
$.get(ROOT_PATH + '/data/asset/data/flare.json', function (data) {
  myChart.hideLoading();
  myChart.setOption(
    (option = {
      tooltip: {
        trigger: 'item',
        triggerOn: 'mousemove'
      },
      series: [
        {
          type: 'tree',
          data: [data],
          top: '18%',
          bottom: '14%',
          layout: 'radial',
          symbol: 'emptyCircle',
          symbolSize: 7,
          initialTreeDepth: 3,
          animationDurationUpdate: 750,
          emphasis: {
            focus: 'descendant'
          }
        }
      ]
    })
  );
});
option && myChart.setOption(option);
</script>
</div>
</div>