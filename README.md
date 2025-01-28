# Whispers_Ollama_FAQ

This is a small bit of info to help address the most common questions I see in the Ollama discord.

## 1) I have a question

Did you read the FAQ yet? [https://github.com/ollama/ollama/blob/main/docs/faq.md](https://github.com/ollama/ollama/blob/main/docs/faq.md)

## 2) Yes, and my question isn't answered by the FAQ

Did you look at the other documentation, such as GPU, API, Docker, Linux, Windows, Troubleshooting, etc?

## 3) Can I run X model?

In order to run a model on Ollama it needs to load the model and some space for message context into memory. Ideally this will be VRAM on your GPU. If the model + context is more than the available VRAM on your GPU it will use your systems RAM.

When you are using RAM in any capacity, it will slow down the models response due to bottlenecks at the speed of the RAM, the CPU, and the Bus.

### 3.1) How do I know what size model I can run?

Look up your GPU, is it supported?  [https://github.com/ollama/ollama/blob/main/docs/gpu.md](https://github.com/ollama/ollama/blob/main/docs/gpu.md)
If yes, how much VRAM does it have?
You will want the size of the model in GBs + 1~2 GBs to be less than your total VRAM.

> ### *for example: `llama3.2:3b` is 2GB in size, this means you need 2GB of VRAM available and 1 to 2 more GBs of VRAM to handle message context, so you should have at least 4GB of VRAM to use it quickly.*
>
> Keep in mind that other things might be using your VRAM as well.

![](images/vram.jpg)



**You CAN use it with RAM, but it will be slower.**


### 3.2) But wait, I have a mac, and it has Unified Memory!

Ok, so basically its the same rules, but don't get hung up on RAM vs VRAM. You still need the memory to run the model.

From what I understand, Unified Memory is slower than VRAM, but faster than RAM. I dunno, I dont use Macs.


## 4) I am using OpenWeb UI and ...

Is it an Ollama question? does the issue happen when using Ollama by itself? If yes, please ask it in the [Ollama discord](https://discord.gg/q7myykKWYR) general chat and/or open a ticket in Help. (But check out the next bullet before hand)

Otherwise, go ask in the OpenWeb UI discord. They will know how to answer OpenWeb UI questions.


## 5) What are Parameters?



## 6) What is Quantization?

LLM quantization is a technique that reduces the precision of weights and activations in large language models (LLMs).

This process makes LLMs more efficient and accessible by reducing their size.

This also makes them less accurate. Generally Q4 and up are fine. Anything smaller can be a little crazy. Mileage May Vary.


## 7) What is Abliteration

Modern LLMs are fine-tuned for safety and instruction-following, meaning they are trained to refuse harmful requests.

In their [blog post](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction), Arditi et al. has shown that this refusal behavior is mediated by a specific direction in the model's residual stream.

If we prevent the model from representing this direction, it  **loses its ability to refuse requests** . Conversely, adding this direction artificially can cause the model to refuse even harmless requests.

source: [https://huggingface.co/blog/mlabonne/abliteration](https://huggingface.co/blog/mlabonne/abliteration)

## 7)

## 5) Ok, I have a question, and none of the previous resources answer it

Great, ok, you are more than welcome to ask it in the Ollama discord general chat or open a ticket in the Help channel. But first, ask yourself a few questions:

1) When I ask the question, am I giving them all the information they need to answer it?
2) How complicated is my question, how much of it depends on my own computer/system?
3) Am I asking someone for help after I have done my own research?
4) When I post my question, do I realize that I might not get an answer right away?

If you are not sure, maybe take a look at how stackoverflow suggests you should ask.

[https://stackoverflow.com/help/how-to-ask](https://stackoverflow.com/help/how-to-ask)

# 6) Lastly! I am just a guy on the internet.

I am not on the Ollama team, I am not a representative of an org. I'm just a guy that likes the community, likes the product, and tries to help people. So if you are offended by anything above, well, I'm sorry for you.

Just keep in mind, these are my opinions. Not a representation of Ollama.
