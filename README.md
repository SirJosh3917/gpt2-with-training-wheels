# GPT2 With Training Wheels

This project is solely for me to get more hands-on with the inner workings of
neural networks. To that effect, I challenged myself to understand GPT-2 and
redo parts of it myself to get a good feel for what I'm doing.

Nothing here is particularly novel, just some basic re-implementation and
explanations of what already exists.

## Getting Started

When looking at older python projects, they always appear to be broken and
require weirdly versioned packages. I'm hoping that this project doesn't fall
victim to the same thing, but I can't be sure since I'm not a python programmer.

Python version: 3.10.9

```sh
# Create a virtual environment to put packages in
python3 -m venv .venv

# Activate it (invoke Activate.ps1 on windows)
. ./.venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download the files needed from the web. Check `main.py`

# Run the program
python3 main.py
```

## Resources

- HuggingFace on tokenization: good overview here and in all the subsequent
  documentation linked. Read as much of it as necessary
  https://huggingface.co/docs/transformers/tokenizer_summary

- GPT-2 Paper
  https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf
- GPT Paper
  https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf
- Transformers Paper
  https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
- Transformers Wikipedia
  https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)

And an assortment of YouTube videos.

Generic explanation videos. Turned my complete lack of knowledge into generic
ideas that I sorta know where they could fit, but definitely still needed more
information to really understand the ideas.

- https://www.youtube.com/watch?v=4Bdc55j80l8
- https://www.youtube.com/watch?v=TQQlZhbC5ps
- https://www.youtube.com/watch?v=XSSTuhyAmnI

And then two other videos:

- **_Phenomenal_** at really diving head-first into transformers specifically,
  wish there was more like it. HIGHLY recommended:
  https://www.youtube.com/watch?v=g2BRIuln4uc
- Not a great explainer, but it's better than no explanation. Helped cover some
  cracks in my knowledge. Terrible "time per thing you learn" ratio, but used it
  nonetheless. https://www.youtube.com/watch?v=kCc8FmEb1nY
