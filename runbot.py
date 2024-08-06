from preparing.chatbot_cpu import Chatbot

bot = Chatbot()

get_command = bot.runbot(
    question="Đoạn văn đã cho được trích trong tác phẩm nào?",
    
    prompt = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra những câu trả lời sai\n{context}<|im_end|>\n<|im_start|>assistant"""
)
print(get_command)
