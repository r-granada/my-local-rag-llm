from src.model.model import Model

my_model = Model()

while True:
    input_question = input("Next Question: ")
    response = my_model.answer(input_question)
    print(f"Response: {response}")

