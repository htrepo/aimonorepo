content = open('main_gradio.py', encoding='utf-8').read()

idx = content.find('final_prompt')

# The old block starts at 'final_prompt' and ends at the '"""' closing (pos 221 relative to idx = idx+221+1)
old_block = content[idx:idx+222]
print("Old block repr:", repr(old_block))

new_block = (
    'final_prompt = (\n'
    '        "Context information is below.\\n"\n'
    '        "---------------------\\n"\n'
    '        f"{context}\\n"\n'
    '        "---------------------\\n"\n'
    '        f"Using only the context above, answer the question: {message}\\n"\n'
    "        \"If the answer is not in the context, say 'I don't know.'\"\n"
    '    )'
)

new_content = content[:idx] + new_block + content[idx+222:]
open('main_gradio.py', 'w', encoding='utf-8').write(new_content)
print("SUCCESS: f-string bug fixed.")
