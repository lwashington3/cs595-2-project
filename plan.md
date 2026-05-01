For the project, the idea will be to use the lottery-ticket hypothesis to reduce the size of LLMs such as LLama3.3.
The hope is that the size of the model (43GB) will be greatly reduced without greatly reducing the accuracy.
To test how efficient the system is, slap it into the EXSCLAIM pipeline and run tests on the original model and on this model.
- Note, a test should be done with each model as is, as well as a test where the models have a new Modelfile that has the system instructions added to it.

Test how accurate the caption separation is when splitting from known articles.


Steps:
1. Figure out how to apply the lottery-ticket hypothesis (Hard part)
   1. Create a repo to hold the project.
   2. Look into the original papers describing this process, as well as some potential methods to do the pruning
   3. Attempt to shrink the model using code
2. Commit all working code to EXSCLAIM, and pull a pinned version of the repo (i.e., commit id) to my repo
   1. Create a test corpora of research articles that are scraped using the JournalScraper and saved.
   2. For each model, run through the UI selecting the tested model and recorded information from /status, as well as memory and independent judging by Tanjin?
3. For the report, make sure to note that I am the current maintainer of EXSCLAIM, as well as any modifications I make to the codebase for this project
   1. In the authorship, include ORCID
   2. In the Appendix, include any short files such as the Modelfiles that were used for training

Articles:
[The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635)
[Successfully Applying the Stabilized Lottery Ticket Hypothesis to the Transformer Architecture](https://arxiv.org/abs/2005.03454)

- Should try doing this on a small LLM first before moving up to larger ones (like Llama 4)
- Also, find a smaller dataset to work with first, before stepping it up

Look into [SafeTensors](https://huggingface.co/docs/safetensors/en/index)
- Can directly read/write to the tensors
- Compatible with [Ollama models](https://docs.ollama.com/modelfile#build-from-a-safetensors-model)
- Might be compatible with PyTorch?
  - Can convert a Pytorch .bin model to .safetensors using their [convert.py](https://github.com/huggingface/safetensors/blob/main/bindings/python/convert.py), but need to make sure its not uploading my models.
    - Will probably need to replace every like that initializes operations with an empty list so it doesn't know to upload it back
- Llama 3.1 has safetensor files on [HuggingFace](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/tree/main)
- Katerina uploaded datasets about EXSCLAIM to [HuggingFace](https://huggingface.co/datasets/kvriza8/microscopy_images)


What I could do is retrain the model, then prune the retrained model and use the dataset I used for retraining for pruning.
This dataset could be related to processing the captions for EXSCLAIM.

Keep reading https://huggingface.co/learn/llm-course/en/chapter3/4

Benchmarking could use [GLUE](https://gluebenchmark.com/), which was recommended by [Hugging Face](https://huggingface.co/learn/llm-course/chapter3/2#loading-a-dataset-from-the-hub)

# Nice to Know:
- What is a model head? An additional component, usually made up of one or a few layers, to convert the transformer predictions to a task-specific output
  - Adaptation heads, also known simply as heads, come up in different forms: language modeling heads, question answering heads, sequence classification heads...

For evaluation, run
- Base model downloaded from HF (as the baseline)
- The full pre-trained model (the one that has yet to be pruned)
- The magnitude-based pruned model
  - Maybe also this model where the values are masked instead of completely removed, testing if the missing structure was obstructing it.
- The structural-pruned model

TODO:
- Find a better dataset that would also cover some niche-LLM task (b/c probably won't generate good EXSCLAIM one in time)
- Implement MRPC if necessary
- Build a function that takes in models to compare and runs them through `LTHPipeline`, before saving the results to a table and saving said table

Models to test:
- https://huggingface.co/Qwen/Qwen3.5-2B
- https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
- https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct