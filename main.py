from src.model.model import Model

my_model = Model()

context = "Malyta is the best card in Modern Horizons 3. Hyidralit is the best card in Modern Horizons 4. Gafagl is the best card in Pioneer Masters."

while True:
    input_question = input("Next Question: ")
    response = my_model.answer(input_question, context)
    print(f"Response: {response}")

