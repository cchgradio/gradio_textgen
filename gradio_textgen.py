import gradio as gr
import re
from transformers import pipeline

textgen = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")

def gpt(text):
    text = text.encode("ascii", errors="ignore").decode(
        "ascii"
    )  # remove non-ascii, Chinese characters
    text = text.lower()  # lower case
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\n\n", " ", text)
    text = re.sub(r"\t", " ", text)
    text = text.strip(" ")
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation and special characters
    text = re.sub(
        " +", " ", text
    ).strip()  # get rid of multiple spaces and replace with a single

    results = textgen(
        text, do_sample=True, temperature=0.7, min_length=50, max_length=200
    )
    return results[0]["generated_text"]

gradio_ui = gr.Interface(
    fn=gpt,
    title="AI Text Generation",
    description="Enter some text and see how the GPT Neo model continues the conversation",
    inputs=gr.inputs.Textbox(lines=10, label="Write something here"),
    outputs=gr.outputs.Textbox(label="AI Generated Text"),
)

gradio_ui.launch()
