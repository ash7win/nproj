print(" guess an animal that is said to laugh! \n you have three guesses \n it helped the villian in lion king")
secret_word= "hyena"
guess= ""
guess_limit= 3
guess_count=0
out_of_guesses= False
while guess != secret_word and not(out_of_guesses):
    if guess_count < guess_limit:
        guess= input("Enter guess: ")
        guess_count += 1
    else:
        out_of_guesses = True

if out_of_guesses:
    print ("Out of guesses! ")
else:
    print ("You win! " )