'''
Written by Austin Walters
Last Edit: January 2, 2019
For use on austingwalters.com

A naive method of classifying sentences
of the common sentance types:
Question, Statement, Command, Exclamation                                                    
'''

from __future__ import print_function
from sentence_types import encode_phrases, gen_test_comments

print("Loading Data...")

comments, comment_categories = gen_test_comments()

question_dict = {
    "who","what","where","when","why","how","which","can", "?"
}

command_dict = {
    "please","don't","shut","fold","open","close","mix","turn",
    "pour","fill","put","add","chop","slice","serve","spread","get",
    "heat","grill","hold","swim","swing","listen","hold","pick",
    "take","fetch","roll","jump","stand","crouch","hide","crack",
    "write","use","order","draw","paint","set","eat","drink","stick",
    "cook","bring","sit","stop","play","buy","shop","explain","tidy",
    "move","switch","improve","behave","sort","go","fly","flip"
}

correct_questions, correct_statements = 0,0
correct_commands, correct            = 0,0
total_questions, total_statements     = 0,0
total_commands, total                = 0,0

for i in range(len(comments)):

    comment  = comments[i].lower()
    category = comment_categories[i].lower()

    classified_category = "statement"
    for word in comment.split(" "):
        if word in question_dict:
            classified_category = "question"
        elif word in command_dict:
            classified_category = "command"
            
    if category == classified_category:
        correct += 1
        if category == "question":
            correct_questions += 1
        elif category == "statement":
            correct_statements += 1
        elif category == "command":
            correct_commands += 1
        
    if category == "question":
        total_questions += 1
    elif category == "statement":
        total_statements += 1
    elif category == "command":
        total_commands += 1        
    total += 1
    
print("Accuracy (questions): %3.2f"
      % (float(correct_questions) / float(total_questions)))
print("Accuracy (statements): %3.2f"
      % (float(correct_statements) / float(total_statements)))
print("Accuracy (commands): %3.2f"
      % (float(correct_commands) / float(total_commands)))
print("Accuracy (overall): %3.2f" % (float(correct) / float(total)))
