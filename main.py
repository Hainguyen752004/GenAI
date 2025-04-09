import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import gradio as gr
import time

# Biến trạng thái toàn cục để theo dõi việc có tiếp tục hay không
should_continue = True

# Load model and tokenizer
def load_model(selected_model):
    if selected_model == "DEEP-TRIET":
        base_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        lora_model_path = "/export/users/1173171/iDragonCloud/Documents/llms/qwen-finetuned/checkpoint-2400"
    elif selected_model == "QWEN-TRIET":
        base_model = "Qwen/Qwen2.5-1.5B"
        lora_model_path = "/export/users/1173171/iDragonCloud/Documents/llms/qwen-no-CoT-finetuned/checkpoint-600"
    else:
        raise ValueError("Invalid model selected")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, trust_remote_code=True)
    model = PeftModel.from_pretrained(model, lora_model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, tokenizer, device

# Function to generate text with cancel check
def generate_text_with_cancel_check(model, tokenizer, device, question):
    global should_continue
    should_continue = True  # Đặt lại trạng thái tiếp tục
    prompt = f"Question: {question}?\nAnswer:"
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    response = ""
    with torch.no_grad():
        for _ in range(100):  # Giả lập vòng lặp dài (có thể thay thế bằng các bước thực tế)
            if not should_continue:
                return "Hủy bỏ xử lý."  # Ngừng xử lý nếu trạng thái đổi
            # Thêm chuỗi vào câu trả lời (mô phỏng)
            response += "Đang xử lý...\n"
            time.sleep(0.1)  # Giả lập trễ, bỏ đi khi sử dụng model.generate thực sự
    return response

# Hàm để xử lý khi nhấn nút "Hủy"
def cancel_processing():
    global should_continue
    should_continue = False
    return "", "Đã hủy."

# CSS for background
css = """
#background-div {
  background-image: url("https://www.danchimviet.info/wp-content/uploads/2021/08/painting-vladimir-putin-karl-marx-joseph-stalin-wallpaper-preview.jpeg");
  background-repeat: no-repeat;
  background-size: cover;
  background-position: center;
  height: 100vh;  /* Chiều cao của div để bao phủ toàn bộ màn hình */
}

#main-content {
  background-color: rgba(255, 255, 255, 0.8); /* Màu nền mờ để làm nổi bật nội dung */
  padding: 20px;
  border-radius: 8px;
}
"""

# Define the interface
def create_interface():
    with gr.Blocks(css=css) as demo:
        gr.HTML('''
        <div id="background-div">
            <iframe src="https://ia800808.us.archive.org/26/items/national-anthem-of-ussr_202503/National%20Anthem%20of%20USSR.mp3?cnt=0" width="1" height="1" frameborder="0" allow="autoplay"></iframe>
        </div>
        ''')
        with gr.Column(elem_id="main-content"):
            gr.Markdown("## 🤖 Hỏi Đáp Triết Học")
            model_choice = gr.Dropdown(["DEEP-TRIET", "QWEN-TRIET"], label="Chọn mô hình", value="DEEP-TRIET")
            question_input = gr.Textbox(placeholder="Nhập câu hỏi của bạn...", lines=1)
            response_output = gr.Textbox(label="Trả lời")
            cancel_button = gr.Button("Hủy")  # Nút hủy

            def handle_input(selected_model, question):
                if not question.strip():
                    return "Vui lòng nhập nội dung câu hỏi!"
                try:
                    model, tokenizer, device = load_model(selected_model)
                    return generate_text_with_cancel_check(model, tokenizer, device, question)
                except Exception as e:
                    return f"Đã xảy ra lỗi: {str(e)}"

            # Gắn sự kiện submit để gửi câu hỏi khi nhấn Enter
            question_input.submit(
                fn=handle_input,
                inputs=[model_choice, question_input],
                outputs=response_output
            )

            # Nút hủy
            cancel_button.click(
                fn=cancel_processing,
                inputs=None,
                outputs=[question_input, response_output]
            )

    return demo

demo = create_interface()
demo.launch()
