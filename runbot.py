from chatbot import Chatbot

bot = Chatbot()

get_command = bot.runbot(
    question="Tác giá của truyện này là ai?",
    
    template = """<|im_start|>system\nSử dụng thông tin sau đây để trả lời câu hỏi. Nếu bạn không biết câu trả lời, hãy nói không biết, đừng cố tạo ra những câu trả lời\n{context}\n<|im_start|>assistant"""
)
print(get_command)
