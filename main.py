import ai

chatmodel = ai.generate.ChatModel()

while True:
    inp = input("You: ")

    if inp in ["exit", "quit"]:
        print("Exiting...")
        break

    output = chatmodel.generate_from_prompt(inp)

    print("AI:", output)