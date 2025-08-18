from bpe.fast_token import FastBPETokenizer
from bpe.byte_encoder import ByteEncoder

from datasets import load_dataset


def clean_text(text):
    if not text or not text.strip():
        return None
    return text.strip()


dataset = load_dataset("stas/openwebtext-10k", split="train")

# Extract just the text column
corpus = [item["text"] for item in dataset]

#Train tokenizer (tiny vocab for demo, 1k)
tokenizer = FastBPETokenizer()
tokenizer.train(corpus[:2000], vocab_size=1000)

sample2=  '''Which tests have 'Pass' results? Return the dates when the tests were taken, and count them by a line chart, and I want to display by the X-axis in asc.
CREATE TABLE Courses (course_id INTEGER,author_id INTEGER,subject_id INTEGER,course_name VARCHAR(120),course_description VARCHAR(255))CREATE TABLE Student_Course_Enrolment (registration_id INTEGER,student_id INTEGER,course_id INTEGER,date_of_enrolment DATETIME,date_of_completion DATETIME)CREATE TABLE Student_Tests_Taken (registration_id INTEGER,date_test_taken DATETIME,test_result VARCHAR(255))CREATE TABLE Students (student_id INTEGER,date_of_registration DATETIME,date_of_latest_logon DATETIME,login_name VARCHAR(40),password VARCHAR(10),personal_name VARCHAR(40),middle_name VARCHAR(40),family_name VARCHAR(40))CREATE TABLE Subjects (subject_id INTEGER,subject_name VARCHAR(120))CREATE TABLE Course_Authors_and_Tutors (author_id INTEGER,author_tutor_ATB VARCHAR(3),login_name VARCHAR(40),password VARCHAR(40),personal_name VARCHAR(80),middle_name VARCHAR(80),family_name VARCHAR(80),gender_mf VARCHAR(1),address_line_1 VARCHAR(80))
=> SELECT date_test_taken, COUNT(date_test_taken) FROM Student_Tests_Taken WHERE test_result = "Pass" ORDER BY date_test_taken
'''
tokens = tokenizer.tokenize(sample2)
print("Tokens:", tokens)

# Detokenization
detok = "".join(tokens).replace("</w>", " ")
print("Detokenized:", detok)


# Token IDs
token_ids = tokenizer.tokenize_to_ids(sample2)
print("Token IDs:", token_ids)

# Decode from IDs
decoded_text = tokenizer.decode_from_ids(token_ids)
print("Decoded Text:", decoded_text)
