import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Define the model name
model_name = "ibm-granite/granite-3.1-2b-base"

# --- Model and Tokenizer Loading ---
# This block is for demonstration. In a production environment,
# you would likely use a more robust way to handle model loading,
# such as a dedicated function or a class.
try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Loading model...")
    # Use torch.bfloat16 for better performance on compatible GPUs
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    # Check for GPU and move model if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model loaded successfully on {device}.")
except Exception as e:
    print(f"Error loading model: {e}")
    tokenizer = None
    model = None

def check_drugs(drug_input):
    """
    Analyzes drug information and potential interactions.
    """
    if not drug_input.strip() or model is None or tokenizer is None:
        return "Please ensure the model is loaded and enter drug names to proceed."
    
    # Create a prompt for the model
    prompt = f"Analyze the following drugs for their uses, side effects, and potential interactions: {drug_input}. Provide a clear, concise summary."
    
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=256, 
            temperature=0.7, 
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and clean the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # This finds the start of the generated text after the prompt.
    start_index = response.find(drug_input) + len(drug_input)
    clean_response = response[start_index:].strip()
    
    if not clean_response:
        return "I couldn't generate a clear analysis for the drugs you provided. Please try a different query."
    
    return clean_response

# Create the Gradio interface
iface = gr.Interface(
    fn=check_drugs, 
    inputs=gr.Textbox(lines=4, label="Enter drug names (e.g., Aspirin, Ibuprofen)"), 
    outputs=gr.Textbox(label="Drug Information and Interactions"),
    title="💊 AI Medical Prescription Verification",
    description="Analyze drug information and potential interactions using the IBM Granite model. This is for informational purposes only and not a substitute for professional medical advice."
)

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch()
