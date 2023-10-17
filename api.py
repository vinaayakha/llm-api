import uvicorn
from fastapi import FastAPI, Request, Response
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-3b-4e1t")
model = AutoModelForCausalLM.from_pretrained(
    "stabilityai/stablelm-3b-4e1t",
    trust_remote_code=True,
    torch_dtype="auto",
)
model.eval()


@app.post("/generate/")
async def generate_text(request: Request):
    data = await request.json()
    text = data.get("text", "")
    inputs = tokenizer(text, return_tensors="pt")
    
    max_length = 64  # Adjust max_length based on your preference
    temperature = 0.6  # Adjust temperature based on your preference
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"generated_text": generated_text}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
