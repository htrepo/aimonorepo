content = open('main_gradio.py', encoding='utf-8').read()

# Find and print the exact bytes in the problem range
idx = content.find('final_prompt')
chunk = content[idx:idx+300]
print("Hex dump of key characters:")
for i, c in enumerate(chunk):
    if c == '"':
        print(f"  pos {i}: DOUBLE QUOTE (0x22)")
    elif c == "'":
        print(f"  pos {i}: SINGLE QUOTE (0x27)")

# Also print the raw repr
print("\nRaw repr:")
print(repr(chunk))
