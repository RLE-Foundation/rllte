---
hide:
  - toc
  - navigation
---

**Copilot** is the first attempt to integrate an LLM into an RL framework, which aims to help developers reduce the learning cost and facilitate application construction. We follow the design of [LocalGPT](https://github.com/PromtEngineer/localGPT) that interacts privately with documents using the power of GPT. The source documents are first ingested by an instructor embedding tool to create a local vector database. After that, a local LLM is used to understand questions and create answers based on the database. In practice, we utilize [Vicuna-7B](https://huggingface.co/lmsys/vicuna-7b-v1.3) as the base model and build the database using various corpora, including API documentation, tutorials, and RL references. The powerful understanding ability of the LLM model enables the copilot to accurately answer questions about the use of the framework and any other questions of RL. Moreover, no additional training is required, and users are free to replace the base model according to their computing power. In future work, we will further enrich the corpus and add the code completion function to build a more intelligent copilot for RL.

- **GitHub Repository**: [https://github.com/RLE-Foundation/rllte-copilot](https://github.com/RLE-Foundation/rllte-copilot)
- **Hugging Face Space**: [Coming soon...]()