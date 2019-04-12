from Question import Question

question_prompts= [
    "Kyun uth thi hai pawan?\n a) Gravity\n b) Solar cycles\n c) Na tum jano na hum\n\n",
    "Kyun machalta hai mann?\n a) Heart disease\n b) Na tum jano na hum\n c) Acidity\n\n",
    "Kyun hota hai nasha?\n a) Na Tum jano na hum\n b) Fever\n c) Drugs\n\n"
]

questions=[
    Question(question_prompts[0],"c"),
    Question(question_prompts[1],"b"),
    Question(question_prompts[2],"c")
]
def run_test(questions):
        score = 0
        for question in questions:
            answer= input(question.prompt)
            if answer == question.answer:
                score += 1
        if score > 2:
            print("Nice!")
        else:
            print("You scored " + str(score) + "/" + str(len(questions)) + " ! you thought it was that easy")

run_test(questions)

