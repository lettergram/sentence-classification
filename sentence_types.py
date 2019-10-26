'''
Written by Austin Walters 
Last Edit: January 2, 2019
For use on https://austingwalters.com

Methods to ingest sentence types
'''

from __future__ import print_function

import os
import re
import numpy as np
import nltk
import json
import random
from bs4 import BeautifulSoup

def encode_phrases(comments, word_encoding=None, add_pos_tags_flag=False):
    '''
    Encodes comments into a vectorized list

    INPUTS:
    :param comments: Array of strings containig contents of comments
    :param word_encoding: Prior hashmap containing mapping of encodings 
    :param add_pos_tags_flag: Flag to add parts-of-speach tags after each word

    RETURNS:
    :return encoded_comments: Comments in a vectorized list of lists
                              words and punctuation vectorized.
    :return word_encoding: hashmap of the word to embeded value
    :return word_decoding: hashmap of the embedded value to word
    '''

    word_decoding = { ' ': 0}     # will store a has to reviews words
    count = 1
    encoded_comments = []

    if word_encoding == None:
        word_encoding = { 0: ' '} # will store a hash of words
    else:
        # Will preload word_decoding if word_encoding exists
        for word in word_encoding:
            word_decoding[word_encoding[word]] = word
        

    for comment in comments:
        encoded_comment = []

        comment = nltk.word_tokenize(comment)

        # Create a POS sentence: word POS_tag word POS_tag, etc.
        if add_pos_tags_flag:
            comment = nltk.pos_tag(comment)
            comment = [ele for word_tuple in comment for ele in word_tuple]
        
        for word in comment:
            word = word.lower() # Lowercase word for mapping

            if word not in word_encoding:
                word_encoding[word] = count
                count += 1
                
            if word not in word_decoding:
                word_decoding[word_encoding[word]] = word
                
            encoded_comment.append(word_encoding[word])
        encoded_comments.append(encoded_comment)
        
    return encoded_comments, word_encoding, word_decoding

def decode_phrases(encoded_comments, word_decoding):
    '''
    Decodes comments to an array of words

    INPUTS:
    :param encoded_comments: Vectorized list of embedded words 
                             and punctiuation. Created by the
                             encode_phrases function
    :param word_decoding: Mapping from word embedding to words.

    RETURNS:
    :return dencoded_comments: Comments in english words (still in list)
    '''
    
    decoded_comments = []
    
    for encoded_comment in encoded_comments:
        decoded_comment = []
        for encoded_word in encoded_comment:
            word = word_decoding[encoded_word]
            decoded_comment.append(word)
        decoded_comments.append(" ".join(decoded_comment))
        
    return decoded_comments

def get_custom_test_comments():
    '''
    Returns a sample test which is easy to manually validates
    '''
    print('\nCreating Manual Test...')
    
    test_comments = [
        "This is a stupid example.",
        "This is another statement, perhaps this will trick the network",
        "I don't understand",
        "What's up?",
        "open the app",
        "This is another example",
        "Do what I tell you",
        "come over here and listen",
        "how do you know what to look for",
        "Remember how good the concert was?",
        "Who is the greatest basketball player of all time?",
        "Eat your cereal.",
        "Usually the prior sentence is not classified properly.",
        "Don't forget about your homework!",
        "Can the model identify a sentence without a question mark",
        "Everything speculated here is VC money and financial bubble with unrelaible financial values. Zomato, uber, paytm, flipkart throw discounts at the rate of losses. May be few can survive at the end. This hurts a lot for SMB too.",
        "I am trying to keep tabs on electric two-wheeler startup industry in India. Ather energy is emerging as a big name. Anyone knows how they are doing?",
        "generally a pretty intuitive way to accomplish a task. Want to trash an app Drag it to the trash Want to print a PDF",
        "Make sure ownership is clear and minimizing opportunities for such problematic outcomes in the second place",
        "Stop the video and walk away."
    ]
    
    test_comments_category = [
        "statement",
        "statement",
        "statement",
        "question",
        "command",
        "statement",
        "command",
        "command",
        "question",
        "question",
        "question",
        "command",
        "statement",
        "command",
        "question",
        "statement",
        "question",
        "question",
        "statement",
        "command"
    ]

    return test_comments, test_comments_category


def gen_test_comments(max_samples=999999999):

    """    
    Generates sample dataset from parsing
    the SQuAD dataset and combining it 
    with the SPAADIA dataset. 

    The data is then shuffled, and two 
    arrays are returned. One contains
    comments the other categories. 

    The indexes for classification 
    in both arrays match, such that
    comment[i] correlates with category[i].

    Types of sentences:

        Statement (Declarative Sentence)
        Question (Interrogative Sentence)
        Exclamation (Exclamatory Sentence)
        Command (Imperative Sentence)

    Current Counts:

         Command: 1264
         Statement: 81104
         Question: 131219

    Ideally, we will improve the command and 
    exclamation data samples to be at least 
    10% of the overall dataset.
    """
    
    
    tagged_comments = {}
    
    with open('data/train-v2.0.json', 'r') as qa:
        parsed = json.load(qa)

    statement_count = 0
    question_count  = 0
    command_count   = 0
        
    # Pulls all data from the SQuAD 2.0 Dataset, adds to our dataset
    for i in range(len(parsed["data"])):
        for j in range(len(parsed["data"][i]["paragraphs"])):
            statements = parsed["data"][i]["paragraphs"][j]["context"]
            if random.randint(0,9) % 4 == 0:                
                statement = statements                
                if statement_count < max_samples and statement not in tagged_comments:
                    tagged_comments[statement] = "statement"
                    statement_count += 1                    
            else:
                
                for statement in statements.split("."):
                    if len(statement) <= 2:
                        continue
                    if random.randint(0,9) % 3 == 0:                        
                        statement += "."
                    if statement_count < max_samples and statement not in tagged_comments:
                        tagged_comments[statement] = "statement"
                        statement_count += 1
                        
            for k in range(len(parsed["data"][i]["paragraphs"][j]["qas"])):
            
                question = parsed["data"][i]["paragraphs"][j]["qas"][k]["question"]
            
                if random.randint(0,9) % 2 == 0:
                    question = question.replace("?", "")
                    
                if random.randint(0,9) % 2 == 0:
                    question = statements.split(".")[0]+". "+question
                    
                if question_count < max_samples and question not in tagged_comments:
                    tagged_comments[question] = "question"                
                    question_count += 1

    # Pulls all data from the SPAADIA dataset, adds to our dataset
    for doc in os.listdir('data/SPAADIA'):
        with open('data/SPAADIA/' + doc, 'r') as handle:
            conversations = BeautifulSoup(handle, features="xml")
            for imperative in conversations.findAll("imp"):
                    imperative = imperative.get_text().replace("\n", "")
                    if command_count < max_samples and imperative not in tagged_comments:
                        tagged_comments[imperative] = "command"
                        command_count += 1
            for declarative in conversations.findAll("decl"):
                    declarative = declarative.get_text().replace("\n", "")
                    if statement_count < max_samples and declarative not in tagged_comments:
                        tagged_comments[declarative] = "statement"
                        statement_count += 1
            for question in conversations.findAll("q-yn"):
                    question = question.get_text().replace("\n", "")
                    if question_count < max_samples and question not in tagged_comments:
                        tagged_comments[question] = "question"
                        question_count += 1
            for question in conversations.findAll("q-wh"):
                    question = question.get_text().replace("\n", "")
                    if question_count < max_samples and question not in tagged_comments:
                        tagged_comments[question] = "question"
                        question_count += 1

    # Pulls all the data from the manually generated imparatives dataset
    with open('data/imperatives.csv', 'r') as imperative_file:
        for row in imperative_file:
            imperative = row.replace("\n", "")
            if command_count < max_samples and imperative not in tagged_comments:
                tagged_comments[imperative] = "command"
                command_count += 1

            # Also add without punctuation
            imperative = re.sub('[^a-zA-Z0-9 \.]', '', row)
            if command_count < max_samples and imperative not in tagged_comments:
                tagged_comments[imperative] = "command"
                command_count += 1
        
    test_comments          = []
    test_comments_category = []

    # Ensure random ordering
    comments = list(tagged_comments.items())
    random.shuffle(comments)

    ###
    ### Balance the dataset
    ###
    local_statement_count = 0
    local_question_count  = 0
    local_command_count   = 0
    
    min_count = min([question_count, statement_count, command_count])

    for comment, category in comments:

        '''
        if category is "statement":
            if local_statement_count > min_count:
                continue
            local_statement_count += 1
        elif category is "question":
            if local_question_count > min_count:
                continue
            local_question_count += 1
        elif category is "command":
            if local_command_count > min_count:
                continue
            local_command_count += 1
        '''
        test_comments.append(comment.rstrip())
        test_comments_category.append(category)

    print("\n-------------------------")
    print("command", command_count)
    print("statement", statement_count)
    print("question", question_count)
    print("-------------------------\n")
        
    return test_comments, test_comments_category


def import_embedding(embedding_name="data/default"):
    '''
    Import word embedding to a giant json document
    '''
    if not embedding_name:
        return None, None
    
    file_flag = os.path.isfile(embedding_name+"_word_encoding.json")
    file_flag &= os.path.isfile(embedding_name+"_cat_encoding.json")
    
    if not file_flag:
        return None, None

    word_encoding = {}
    with open(embedding_name+"_word_encoding.json") as word_embedding:
        word_encoding = json.load(word_embedding)

    category_encoding = {}
    with open(embedding_name+"_cat_encoding.json") as cat_embedding:
        category_encoding = json.load(cat_embedding)
    
    return word_encoding, category_encoding
    

def export_embedding(word_encoding, category_encoding,
                     embedding_name="data/default"):
    '''
    Export word embedding to a giant json document
    '''
    if not embedding_name \
       or (not word_encoding) or 2 > len(word_encoding) \
       or (not category_encoding) or 2 > len(category_encoding):
        return
    
    with open(embedding_name+"_word_encoding.json", "w") as embedding:
        embedding.write(json.dumps(word_encoding))

    with open(embedding_name+"_cat_encoding.json", "w") as embedding:
        embedding.write(json.dumps(category_encoding))
    
    
def encode_data(test_comments, test_comments_category,
                data_split=0.8, embedding_name=None, add_pos_tags_flag=False):
    
    print("Encoding Data...")
    
    # Import prior mapping
    word_encoding, category_encoding = None, None
    if embedding_name:
        word_encoding, category_encoding = import_embedding(embedding_name)

    # Encode comments word + punc, using prior mapping or make new
    encoded_comments, word_encoding, \
        word_decoding = encode_phrases(test_comments, word_encoding,
                                       add_pos_tags_flag=add_pos_tags_flag)
    
    encoded_categories, categories_encoding, \
        categories_decoding = encode_phrases([" ".join(test_comments_category)],
                                             category_encoding,
                                             add_pos_tags_flag=False)

    print("Embedding Name", embedding_name)
    if embedding_name:
        export_embedding(word_encoding, categories_encoding,
                         embedding_name=embedding_name)
    
    training_sample = int(len(encoded_comments) * data_split)
    
    print(np.array(encoded_categories[0]))

    x_train = np.array(encoded_comments[:training_sample])
    x_test  = np.array(encoded_comments[training_sample:])
    y_train = np.array(encoded_categories[0][:training_sample])
    y_test  = np.array(encoded_categories[0][training_sample:])

    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    return x_train, x_test, y_train, y_test


def load_encoded_data(data_split=0.8, embedding_name="data/default", pos_tags=False):
    '''
    Loads and encodes embeddings
    If data has already been encoded, uses the pre-encoded data
    If data has not been encoded, encodes, then saves for future use
    returns split training and testing data
    '''

    file_name = embedding_name+"_pre_encoded_comments.csv"
    if pos_tags:
        file_name = embedding_name+"_pre_encoded_pos_tagged_comments.csv"

    encoded_comments   = []
    encoded_categories = []

    # If the encoded results are not available for quick load,
    # create a cache in the data folder of encoded data to pull quickly
    if not os.path.isfile(file_name):

        print("No Cached Data Found...")
        print("Loading Data...")
        
        test_comments, test_comments_category = gen_test_comments()        
        x_train, x_test, y_train, y_test = encode_data(test_comments,
                                                       test_comments_category,
		                                       data_split=data_split,
                                                       embedding_name="data/default",
                                                       add_pos_tags_flag=pos_tags)
        for i in range(len(y_train)):
            encoded_comments.append([y_train[i], x_train[i]])
        for i in range(len(y_test)):
            encoded_comments.append([y_test[i], x_test[i]])

        with open(file_name, 'w') as encoding_file:
            for row in encoded_comments:
                encoding_file.write(str(row[0]) + "|||||" + str(row[1]))
                encoding_file.write('\n')
    else:
        print("Loading Data...")
                
    encoded_comments   = []
    encoded_categories = []
    with open(file_name, 'r') as encoding_file:
        for row in encoding_file:
            row = row.split("|||||")
            row[0] = int(row[0])
            row[1] = row[1].replace("\n", "").replace("[", "").replace("]", "")
            row[1] = np.array(row[1].split(",")).astype('int').tolist()
            encoded_categories.append(row[0])
            encoded_comments.append(row[1])
                
    training_sample = int(len(encoded_comments) * data_split)

    x_train = np.array(encoded_comments[:training_sample])
    x_test  = np.array(encoded_comments[training_sample:])
    y_train = np.array(encoded_categories[:training_sample])
    y_test  = np.array(encoded_categories[training_sample:])

    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    return x_train, x_test, y_train, y_test

