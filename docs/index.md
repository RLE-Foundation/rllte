---
hide:
  - toc
---

# **RLLTE: Long-Term Evolution Project of Reinforcement Learning**



---

Inspired by the long-term evolution (LTE) standard project in telecommunications, aiming to provide development components and standards for advancing RL research and applications. Beyond delivering top-notch algorithm implementations, **RLLTE** also serves as a **toolkit** for developing algorithms.

<div align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/PMF6fa72bmE?si=oDLvQqxVrMP31Iqk" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
<br>
An introduction to RLLTE.
</div>

## **Why RLLTE?**
- 🧬 Long-term evolution for providing latest algorithms and tricks;
- 🏞️ Complete ecosystem for task design, model training, evaluation, and deployment (TensorRT, CANN, ...);
- 🧱 Module-oriented design for complete decoupling of RL algorithms;
- 🚀 Optimized workflow for full hardware acceleration;
- ⚙️ Support custom environments and modules;
- 🖥️ Support multiple computing devices like GPU and NPU;
- 💾 Large number of reusable benchmarks (See [rllte-hub](https://hub.rllte.dev));
- 👨‍✈️ Large language model-empowered copilot.

## **A `PyTorch` for RL**
RLLTE decouples RL algorithms into minimum primitives and provide standard modules for development. 

See [Fast Algorithm Development]() for detailed examples.
<div align=left>
<img src='./assets/images/structure.svg' style="width: 80%">
</div>


## **Project Evolution**
**RLLTE** selects RL algorithms based on the following tenet:

- Generality is the most important;
- Improvements in sample efficiency or generalization ability;
- Excellent performance on recognized benchmarks;
- Promising tools for RL.

## **Cite Us**
If you use **RLLTE** in your research, please cite this project like this:
```bibtex
@article{yuan2023rllte,
  title={RLLTE: Long-Term Evolution Project of Reinforcement Learning}, 
  author={Mingqi Yuan and Zequn Zhang and Yang Xu and Shihao Luo and Bo Li and Xin Jin and Wenjun Zeng},
  year={2023},
  journal={arXiv preprint arXiv:2309.16382}
}
```