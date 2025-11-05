import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = ""  # change to your fine-tuned Qwen path or huggingface repo

def load_model_and_tokenizer():
    print(f"Loading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return model, tokenizer

def extract_fields_with_model(model, tokenizer, text):
    prompt = f"Extract structured fields as JSON from the following text:\n\n{text}\n\nJSON:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.2)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        json_part = response.split("JSON:")[-1].strip()
        parsed = json.loads(json_part)
    except Exception:
        parsed = {"raw_output": response}
    return parsed

if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer()
    sample_text = """
الموظف:
أهلاً بحضرتك، خدمة العملاء، إزاي أقدر أساعدك؟

العميل:
مساء الخير، أنا عندي مشكلة في الفاتورة الشهرية، المبلغ أعلى من المعتاد.

الموظف:
تمام يا فندم، ممكن آخد رقم الحساب أو الخط علشان أراجع؟

العميل:
آه طبعًا، الرقم 0101234567.

الموظف:
لحظة واحدة... (بيراجع البيانات)
تمام يا فندم، واضح إن فيه خدمة اتفعلت بالخطأ الشهر اللي فات.

العميل:
خدمة إيه؟ أنا ما طلبتش حاجة جديدة.

الموظف:
هي خدمة باقة إضافية للإنترنت. ممكن يكون اتفعلت بالغلط. أنا أقدر أوقفها دلوقتي وأخصم المبلغ في الفاتورة الجاية.

العميل:
طب تمام، ياريت توقفها فورًا.

الموظف:
اتلغت خلاص يا فندم. وهيوصلك رسالة تأكيد دلوقتي. في أي استفسار تاني ممكن أساعدك فيه؟

العميل:
لا، كده تمام، شكراً جدًا.

الموظف:
العفو يا فندم، تحت أمرك في أي وقت. يومك سعيد.

"""
    fields = extract_fields_with_model(model, tokenizer, sample_text)
    print(json.dumps(fields, indent=2))
