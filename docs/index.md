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
<script src="./assets/stylesheets/echarts.js"></script>
<div id="main" style="width: 750px; height:400px;">
    <script type="text/javascript">
    $.getJSON("./assets/images/structure.json", function(data) {});
      var myChart = echarts.init(document.getElementById('main'));
      var option = {
        tooltip: {
            trigger: 'item',
            triggerOn: 'mousemove'
            },
        series: [{
            type: 'tree',
            data: [data],
            top: '1%',
            left: '7%',
            bottom: '1%',
            right: '20%',
            symbolSize: 7,
            label: {
                position: 'left',
                verticalAlign: 'middle',
                align: 'right',
                fontSize: 9
                },
        leaves: {
            label: {
            position: 'right',
            verticalAlign: 'middle',
            align: 'left'}
            },
        emphasis: {
            focus: 'descendant'
            },
        expandAndCollapse: true,
        animationDuration: 550,
        animationDurationUpdate: 750
        }]
      };
      myChart.setOption(option);
    </script>
</div>
</div>