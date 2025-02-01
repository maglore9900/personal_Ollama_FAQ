# Whispers_Ollama_FAQ

This is a small bit of info to help address the most common questions I see in the Ollama discord.

Ollama Questions
* [1) I have a question](#1-i-have-a-question)
* [2) It fails when I pull / run a model](#2-it-fails-when-i-pull--run-a-model)
  + [2.1) Pulling a model?](#21-pulling-a-model)
  + [2.2) Running a model?](#22-running-a-model)
* [3) Can I run X model?](#3-can-i-run-x-model)
  + [3.1) How do I know what size model I can run?](#31-how-do-i-know-what-size-model-i-can-run)
  + [3.2) I don't have a GPU...](#32-i-dont-have-a-gpu)
  + [3.3) I have more than one GPU...](#33-i-have-more-than-one-gpu)
  + [3.4) But wait, I have a mac, and it has Unified Memory!](#34-but-wait-i-have-a-mac-and-it-has-unified-memory)
* [4) Ollama is not using all my resources OR GPU is not being used but my CPU is at 100%](#4-ollama-is-not-using-all-my-resources-or-gpu-is-not-being-used-but-my-cpu-is-at-100)
* [5) I am using OpenWeb UI and ...](#5-i-am-using-openweb-ui-and)
* [6) Ok, I have a question, and none of the previous resources answer it](#9-ok-i-have-a-question-and-none-of-the-previous-resources-answer-it)

Further Reading
*  [1) What are Parameters?](#6-what-are-parameters)
* [2) What is Quantization?](#7-what-is-quantization)
* [3) What is Abliteration?](#8-what-is-abliteration)
* [4) What is Embedding?](#10-what-is-embedding)
* [5) What is Retrieval Augmented Generation (RAG)](#5-what-is-retrieval-augmented-generation-rag)
  + [5.1) So what do we do then?](#51-so-what-do-we-do-then)
  + [5.2) Check out these resources!](#52-check-out-these-resources)
* [6) What is Fine-tuning?(#6-what-is-fine-tuning)
* [7) Lastly! I am just a guy on the internet.](#12-lastly-i-am-just-a-guy-on-the-internet)


# Ollama Questions

## 1) I have a question

Did you read the FAQ yet? [https://github.com/ollama/ollama/blob/main/docs/faq.md](https://github.com/ollama/ollama/blob/main/docs/faq.md)

### 1.1) Yes, and my question isn't answered by the FAQ

Did you look at the other documentation, such as GPU, API, Docker, Linux, Windows, Troubleshooting, etc?

## 2) It fails when I pull / run a model

### 2.1) Pulling a model?

Try changing your DNS to 1.1.1.1 or 8.8.8.8

The most common issue is caused by some kind of DNS caching.

### 2.2) Running a model?

Make sure you have the most up to date version of Ollama.

Make sure you installed it using the methods shown on the Ollama website.

Still having an issue? Go to the Ollama discord and open a ticket in the #Help channel. See #9 below.

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

### 3.2) I don't have a GPU...
That's fine, Ollama automatically will use your RAM. Just in case you missed it all the times its written here, RAM IS SLOW. 

# 3.3) I have more than one GPU...
Ollama will use more than one GPU, BUT:
- They need to be supported
- They need to be the same manufacturer (both Nvidia or Both AMD, etc)
*Caveat: You can run multiple instances of Ollama, one on an Nvidia card and one on an AMD for example*

## 3.3) But wait, I have a mac, and it has Unified Memory!

Ok, so basically its the same rules, but don't get hung up on RAM vs VRAM. You still need the memory to run the model.

From what I understand, Unified Memory is slower than VRAM, but faster than RAM. I dunno, I dont use Macs.

## 4) Ollama is not using all my resources OR GPU is not being used but my CPU is at 100%

So, if you read #3 you know that you really want Ollama to use your GPU and to do that the model needs to fit into the available VRAM.

If you are experiencing this problem, you probably have a model that doesnt fit OR you dont have a supported GPU.

So why do you see all the CPU usage and nothing on GPU.

The answer is simple. The GPU is waiting for the CPU to catch up. This is because the CPU, RAM, and Bus are slower. One, if not all three, of those will be the constraint.

There is no magic work around for this, except you may find you get a bit better performance if you choose an LLM that is optimzied for CPU, like the Granite3 models.

## 5) I am using OpenWeb UI and ...

Is it an Ollama question? does the issue happen when using Ollama by itself? If yes, please ask it in the [Ollama discord](https://discord.gg/q7myykKWYR) general chat and/or open a ticket in Help. (But check out the #9 before hand)

Otherwise, go ask in the OpenWeb UI discord. They will know how to answer OpenWeb UI questions.

## 6) Ok, I have a question, and none of the previous resources answer it

Great, ok, you are more than welcome to ask it in the Ollama discord general chat or open a ticket in the Help channel. But first, ask yourself a few questions:

1) When I ask the question, am I giving them all the information they need to answer it?
2) How complicated is my question, how much of it depends on my own computer/system?
3) Am I asking someone for help after I have done my own research?
4) When I post my question, do I realize that I might not get an answer right away?

If you are not sure, maybe take a look at how stackoverflow suggests you should ask.

[https://stackoverflow.com/help/how-to-ask](https://stackoverflow.com/help/how-to-ask)



# Further Reading


## 1) What are Parameters?

LLM parameters are numerical values that control how a Large Language Model (LLM) processes and generates text.

They are learned during training and adjusted to help the model understand language.

More is technically better, but in a lot of cases the extra accuracy is not really usable.

## 2) What is Quantization?

LLM quantization is a technique that reduces the precision of weights and activations in large language models (LLMs).

This process makes LLMs more efficient and accessible by reducing their size.

This also makes them less accurate. Generally Q4 and up are fine. Anything smaller can be a little crazy. **Mileage May Vary.**

![](images/quant.png)

file: [quantization_type.xlsx](quantization_type.xlsx)

## 3) What is Abliteration?

Modern LLMs are fine-tuned for safety and instruction-following, meaning they are trained to refuse harmful requests.

In their [blog post](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction), Arditi et al. has shown that this refusal behavior is mediated by a specific direction in the model's residual stream.

If we prevent the model from representing this direction, it  **loses its ability to refuse requests** .

Conversely, adding this direction artificially can cause the model to refuse even harmless requests.

## 4) What is Embedding?

In the context of Large Language Models (LLMs), an "embedding" is a numerical representation of a word, phrase, or piece of text that captures its semantic meaning , allowing the LLM to understand the context and relationships between different pieces of information by placing them as vectors in a high-dimensional space where similar concepts are positioned close together; essentially, it's a way to translate language into a format that computers can easily process and reason with

check out this article: [https://medium.com/mongodb/how-to-choose-the-best-embedding-model-for-your-llm-application-2f65fcdfa58d](https://medium.com/mongodb/how-to-choose-the-best-embedding-model-for-your-llm-application-2f65fcdfa58d)

## 5) What is Retrieval Augmented Generation (RAG)

Retrieval Augmented Generation (RAG) is  an AI technique that enhances the capabilities of a large language model (LLM) by allowing it to access and reference external information sources, like a knowledge base or database, before generating a response, essentially providing more context and accuracy to the generated output by retrieving relevant data specific to a given query or situation.

The simplest form of RAG is taking the content of a document and simply shoving it into the context windows of an LLM. The fact that LLMs can have pretty large context windows these days means this is somewhat feasible. However, its not a great solution.

First, you are giving it a lot of data, and to fit that you need a large context window, so that's going to eat a lot of your hardware's available memory. This can cause the LLMs to slow down (see #3), or you may have more data than what can fit in the context window.

Second, its going to take longer because there is a lot of data that's not relevant to your query.

Third, many LLMs will find the first close enough match and respond. What if the comprehensive answer can only be provided when all of the information has been reviewed.

### 5.1) So what do we do then?

Check out the document links below, but the TLDR; is that you will use an LLM like a retriever to find the most relevant / top k results, and provide those to the LLM to answer. (Honestly at a minimum I would suggest this). Then you are more likely to get a comprehensive answer, and its much quicker, and a more efficient use of your resources.

As you dig into the links below you will see that it can get pretty advanced, and you may find that some solutions are not necessary for your use case. Thats fine, better to know what you can do and not need it, than the alternative.

### 5.2) Check out these resources!

- Intro to **RAG**: [https://blog.gopenai.com/**rag**-in-action-enhancing-ai-with-real-time-data-retrieval-9fc216710013](https://blog.gopenai.com/rag-in-action-enhancing-ai-with-real-time-data-retrieval-9fc216710013)
- Bit more advanced, i wrote this one: [https://medium.com/@sergio1101102/mastering-retrieval-augmented-generation-**rag**-a-practical-guide-for-new-developers-624be24ca516](https://medium.com/@sergio1101102/mastering-retrieval-augmented-generation-rag-a-practical-guide-for-new-developers-624be24ca516 "https://medium.com/@sergio1101102/mastering-retrieval-augmented-generation-rag-a-practical-guide-for-new-developers-624be24ca516")
- A simple **RAG** tool that uses ollama, you can copy the code for what you need: [https://github.com/maglore9900/chat_with_docs](https://github.com/maglore9900/chat_with_docs "https://github.com/maglore9900/chat_with_docs")
- Bit even more advanced, where I wrote about my own experience with a specific use case: [https://medium.com/@sergio1101102/mastering-**rag**-a-practical-guide-for-new-developers-part-2-786858742e91](https://medium.com/@sergio1101102/mastering-rag-a-practical-guide-for-new-developers-part-2-786858742e91 "https://medium.com/@sergio1101102/mastering-rag-a-practical-guide-for-new-developers-part-2-786858742e91")

source: [https://huggingface.co/blog/mlabonne/abliteration](https://huggingface.co/blog/mlabonne/abliteration)

## 6) What is Fine-tuning?

Before training an LLM/fine-tuning it on any data, question yourself is there a need for training it?

Now, if your data is quite large, fine-tuning would be a good way only if the data is arranged in Q/A (question-answer) format. If the data is unorganized don't waste your time/money organizing it. If your data increases gradually/isn't fixed, fine-tuning/training isn't a thing for you. If your data is unorganized and isn't fixed, you should go for retrieval augmented generation instead. In it, we find the most similar chunks of text (from the data) that is relevent to the question and pass the data to the LLM and ask it to generate an answer on the basis of that.

Even after reading this, if you'd like to train/fine-tune checkout Unsloth AI and if you changed your mind and want to know more about retrieval augmented generation checkout this blog: https://js.langchain.com/docs/concepts/rag

source: [https://github.com/ItzCrazyKns](https://github.com/ItzCrazyKns)

## 7) Lastly! I am just a guy on the internet.

I am not on the Ollama team, I am not a representative of an org. I'm just a guy that likes the community, likes the product, and tries to help people. So if you are offended by anything above, well, I'm sorry for you.

Just keep in mind, these are my opinions. Not a representation of Ollama.