content = open('main_gradio.py', encoding='utf-8').read()

old = 'final_prompt = f"""Context information is below.\\n---------------------\\n{context}\\n---------------------\\nUsing only the context above, answer the question: {message}\\nIf the answer is not in the context, say "I don\'t know.\\""""'

new = (
    'final_prompt = (\n'
    '        "Context information is below.\\n"\n'
    '        "---------------------\\n"\n'
    '        f"{context}\\n"\n'
    '        "---------------------\\n"\n'
    '        f"Using only the context above, answer the question: {message}\\n"\n'
    "        \"If the answer is not in the context, say \\\"I don't know.\\\"\"\n"
    '    )'
)

if old in content:
    content = content.replace(old, new)
    open('main_gradio.py', 'w', encoding='utf-8').write(content)
    print("SUCCESS: f-string bug fixed.")
else:
    print("ERROR: pattern not found, printing surrounding context...")
    idx = content.find('final_prompt')
    print(repr(content[idx:idx+300]))
