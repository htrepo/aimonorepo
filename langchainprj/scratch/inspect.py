content = open('main_gradio.py', encoding='utf-8').read()
idx = content.find('final_prompt')
print(repr(content[idx:idx+350]))
