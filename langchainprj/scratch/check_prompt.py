context = "MaryAnn is being paid 1M dollars for her job."
message = "what is maryann salary"

final_prompt = (
    "Context information is below.\n"
    "---------------------\n"
    f"{context}\n"
    "---------------------\n"
    f"Using only the context above, answer the question: {message}\n"
    "If the answer is not in the context, say 'I don't know.'"
)

print("=== EXACT PROMPT SENT TO LLM ===")
print(final_prompt)
print("=================================")
