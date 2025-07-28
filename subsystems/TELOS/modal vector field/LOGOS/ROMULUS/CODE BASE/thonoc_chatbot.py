# thonoc_chatbot.py

import gradio as gr
from thonoc_engine import ThonocProofEngine

engine = ThonocProofEngine()

def handle_input(query: str) -> str:
    result = engine.process(query)
    return result["summary"]

def handle_full_response(query: str) -> dict:
    result = engine.process(query)
    e, g, t = result["trinity"]
    c = result["fractal"]["c"]
    return {
        "Summary": result["summary"],
        "Lambda Expression": result["lambda_expr"],
        "Trinity Vector": f"𝔼: {e:.2f}, 𝔾: {g:.2f}, 𝕋: {t:.2f}",
        "Modal Status": result["modal_status"].capitalize(),
        "Fractal c": f"{c.real:.3f} + {c.imag:.3f}i",
        "Mandelbrot Set": "✅ In set" if result["fractal"]["in_set"] else f"❌ Escaped after {result['fractal']['iterations']} iterations",
        "Coherence": f"{result['coherence']:.2f}"
    }

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 🧙‍♂️ THŌNOC: The Logos Sage\nAsk anything. Receive metaphysical insight.\n")

    with gr.Row():
        query_input = gr.Textbox(label="Your Question", placeholder="e.g. Is goodness dependent on truth?")
        basic_output = gr.Textbox(label="Sage Summary", interactive=False)

    ask_button = gr.Button("Ask the Sage")

    with gr.Accordion("🧬 Full Inference Details", open=False):
        full_output = gr.JSON(label="Full Reasoning")

    ask_button.click(fn=handle_input, inputs=[query_input], outputs=[basic_output])
    ask_button.click(fn=handle_full_response, inputs=[query_input], outputs=[full_output])

if __name__ == "__main__":
    demo.launch()
