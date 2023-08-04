# RL Algorithm Decoupling

The actual performance of an RL algorithm is affected by various factors (e.g., different network architectures and experience usage 
strategies), which are difficult to quantify.

> Huang S, Dossa R F J, Raffin A, et al. The 37 Implementation Details of Proximal Policy Optimization[J]. The ICLR Blog Track 2023, 2022.

**RLLTE** decouples RL algorithms into minimum primitives from the perspective of exploitation and exploration and provides abundant modules for development:

- **Xploit**: Modules that focus on exploitation in RL.
    - **Encoder**: Modules for processing observations and extracting features;
    - **Policy**: Modules for interaction and learning;
    - **Storage**: Modules for storing and replaying collected experiences;
- **Xplore**: Modules that focus on exploration in RL.
    - **Distribution**: Modules for sampling actions;
    - **Augmentation**: Modules for observation augmentation;
    - **Reward**: Intrinsic reward modules for enhancing exploration.

Therefore, the core of **RLLTE** is not designed to provide specific RL algorithms but a toolkit for producing algorithms. Developers are free to use various built-in or customized modules to build RL algorithms.

- > See [Fast Algorithm Development](../development.md)

In particular, developers are allowed to replace modules of an implemented algorithm.

- > See [Module Replacement for An Implemented Algorithm](../../mt/replacement.md)

**RLLTE** is an extremely open framework that allows developers to try anything.