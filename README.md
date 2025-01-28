# Whispers_Ollama_FAQ

This is a small bit of info to help address the most common questions I see in the Ollama discord.

## 1) I have a question

Did you read the FAQ yet? [https://github.com/ollama/ollama/blob/main/docs/faq.md](https://github.com/ollama/ollama/blob/main/docs/faq.md)

## 2) Yes, and my question isn't answered by the FAQ

Did you look at the other documentation, such as GPU, API, Docker, Linux, Windows, Troubleshooting, etc?

## 3) Can I run X model?

In order to run a model on Ollama it needs to load the model and some space for message context into memory.

Ideally this will be VRAM on your GPU. If the model + context is more than the available VRAM on your GPU it will use your systems RAM.

When you are using RAM in any capacity, it will slow down the models response due to bottlenecks at the speed of the RAM, the CPU, and the Bus.

### 3.1) How do I know what size model I can run?

Look up your GPU, is it supported?  [https://github.com/ollama/ollama/blob/main/docs/gpu.md](https://github.com/ollama/ollama/blob/main/docs/gpu.md)

If yes, how much VRAM does it have?

You will want the size of the model in GBs + 1~2 GBs to be less than your total VRAM.

> ### *for example: `llama3.2:3b` is 2GB in size, this means you need 2GB of VRAM available and 1 to 2 more GBs of VRAM to handle message context, so you should have at least 4GB of VRAM to use it quickly.*
>
> Keep in mind that other things might be using your VRAM as well.

![](images/vram.jpg)

*You CAN use it with RAM, but it will be slower.*

### 3.2) But wait, I have a mac, and it has Unified Memory!

Ok, so basically its the same rules, but don't get hung up on RAM vs VRAM. You still need the memory to run the model.

From what I understand, Unified Memory is slower than VRAM, but faster than RAM. I dunno, I dont use Macs.

## 4) I am running Ollama but its not using all my resources OR I have a GPU and its not being used but my CPU is at 100%

So, if you read #3 you know that you really want Ollama to use your GPU and to do that the model needs to fit into the available VRAM.

If you are experiencing this problem, you probably have a model that doesnt fit OR you dont have a supported GPU.

So why do you see all the CPU usage and nothing on GPU.

The answer is simple. The GPU is waiting for the CPU to catch up. This is because the CPU, RAM, and Bus are slower. One if not all three of those will be the constraint.

There is no magic work around for this, except you may find you get a bit better performance if you choose an LLM that is optimzied for CPU, like the Granite3 models.

## 5) I am using OpenWeb UI and ...

Is it an Ollama question? does the issue happen when using Ollama by itself? If yes, please ask it in the [Ollama discord](https://discord.gg/q7myykKWYR) general chat and/or open a ticket in Help. (But check out the next bullet before hand)

Otherwise, go ask in the OpenWeb UI discord. They will know how to answer OpenWeb UI questions.

## 6) What are Parameters?

LLM parameters are numerical values that control how a Large Language Model (LLM) processes and generates text.

They are learned during training and adjusted to help the model understand language.

## 7) What is Quantization?

LLM quantization is a technique that reduces the precision of weights and activations in large language models (LLMs).

This process makes LLMs more efficient and accessible by reducing their size.

This also makes them less accurate. Generally Q4 and up are fine. Anything smaller can be a little crazy. **Mileage May Vary.**

![](images/quant.png)

file: [quantization_type.xlsx](quantization_type.xlsx)

## 8) What is Abliteration

Modern LLMs are fine-tuned for safety and instruction-following, meaning they are trained to refuse harmful requests.

In their [blog post](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction), Arditi et al. has shown that this refusal behavior is mediated by a specific direction in the model's residual stream.

If we prevent the model from representing this direction, it  **loses its ability to refuse requests** .

Conversely, adding this direction artificially can cause the model to refuse even harmless requests.

source: [https://huggingface.co/blog/mlabonne/abliteration](https://huggingface.co/blog/mlabonne/abliteration)

## 9) Ok, I have a question, and none of the previous resources answer it

Great, ok, you are more than welcome to ask it in the Ollama discord general chat or open a ticket in the Help channel. But first, ask yourself a few questions:

1) When I ask the question, am I giving them all the information they need to answer it?
2) How complicated is my question, how much of it depends on my own computer/system?
3) Am I asking someone for help after I have done my own research?
4) When I post my question, do I realize that I might not get an answer right away?

If you are not sure, maybe take a look at how stackoverflow suggests you should ask.

[https://stackoverflow.com/help/how-to-ask](https://stackoverflow.com/help/how-to-ask)

## 10) What is embedding?

In the context of Large Language Models (LLMs), an "embedding" is a numerical representation of a word, phrase, or piece of text that captures its semantic meaning , allowing the LLM to understand the context and relationships between different pieces of information by placing them as vectors in a high-dimensional space where similar concepts are positioned close together; essentially, it's a way to translate language into a format that computers can easily process and reason with

check out this article: [https://medium.com/mongodb/how-to-choose-the-best-embedding-model-for-your-llm-application-2f65fcdfa58d](https://medium.com/mongodb/how-to-choose-the-best-embedding-model-for-your-llm-application-2f65fcdfa58d)


## 11) What is Retrieval Augmented Generation (RAG)

Retrieval Augmented Generation (RAG) is  an AI technique that enhances the capabilities of a large language model (LLM) by allowing it to access and reference external information sources, like a knowledge base or database, before generating a response, essentially providing more context and accuracy to the generated output by retrieving relevant data specific to a given query or situation.

The simpliest form of RAG is taking the content of a document and simply shoving it into the context windows of an LLM. The fact that LLMs can have pretty large context windows these days means this is somewhat feasible. However, its not a great solution.

First, you are giving it a lot of data, and to fit that you need a large context window, so thats going to eat a lot of your hardwares available memory. This can cause the LLMs to slow down (see #3), or you may have more data than what can fit in the context window.

Second, its going to take longer because there is alot of data thats not relevant to your query.

Third, many LLMs will find the first close enough match and respond. What if the comprehensive answer can only be provided when all of the information has been reviewed.

### 11.1) So what do we do then?

Check out the document links below, but the TLDR; is that you will use an LLM like a retriever to find the most relevant / top k results, and provide those to the LLM to answer. (Honestly at a minimum I would suggest this). Then you are more likely to get a comprehensive answer, and its much quicker, and a more efficient use of your resources.

As you dig into the links below you will see that it can get pretty advanced, and you may find that some solutions are not necessary for your use case. Thats fine, better to know what you can do and not need it, than the alternative.

### 11.2) Check out these resources!

- Intro to **RAG**: [https://blog.gopenai.com/**rag**-in-action-enhancing-ai-with-real-time-data-retrieval-9fc216710013](https://blog.gopenai.com/rag-in-action-enhancing-ai-with-real-time-data-retrieval-9fc216710013"https://blog.gopenai.com/rag-in-action-enhancing-ai-with-real-time-data-retrieval-9fc216710013")
- Bit more advanced, i wrote this one: [https://medium.com/@sergio1101102/mastering-retrieval-augmented-generation-**rag**-a-practical-guide-for-new-developers-624be24ca516](https://medium.com/@sergio1101102/mastering-retrieval-augmented-generation-rag-a-practical-guide-for-new-developers-624be24ca516 "https://medium.com/@sergio1101102/mastering-retrieval-augmented-generation-rag-a-practical-guide-for-new-developers-624be24ca516")
- A simple **RAG** tool that uses ollama, you can copy the code for what you need: [https://github.com/maglore9900/chat_with_docs](https://github.com/maglore9900/chat_with_docs "https://github.com/maglore9900/chat_with_docs")
- Bit even more advanced, where I wrote about my own experience with a specific use case: [https://medium.com/@sergio1101102/mastering-**rag**-a-practical-guide-for-new-developers-part-2-786858742e91](https://medium.com/@sergio1101102/mastering-rag-a-practical-guide-for-new-developers-part-2-786858742e91 "https://medium.com/@sergio1101102/mastering-rag-a-practical-guide-for-new-developers-part-2-786858742e91")


## 12) Lastly! I am just a guy on the internet.

I am not on the Ollama team, I am not a representative of an org. I'm just a guy that likes the community, likes the product, and tries to help people. So if you are offended by anything above, well, I'm sorry for you.

Just keep in mind, these are my opinions. Not a representation of Ollama.
