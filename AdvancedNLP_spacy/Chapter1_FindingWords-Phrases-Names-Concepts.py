# Chapter 1: Finding words, phrases, names and concepts
# This chapter will introduce you to the basics of text processing with spaCy. 
# You'll learn about the data structures, how to work with trained pipelines, and how to use them to predict linguistic features in your text
import spacy
from helpermethods import sectionheader

sectionheader("getting started")
# there are different languages you can use (en, de, es)
nlp = spacy.blank("en")

# Processing the doc is as simple as this one line
# this practice sentenct needs to be at least ten words
doc = nlp("This is my practice text for testing. I am enjoying learning python and spacy")


# Print the text
print(doc.text)


# Docs contain "tokens" and "spans"
# the following is an example of how to access a token and print it's text
first_token = doc[0]
print(first_token.text)


# Creating Slices of docs is just like a list slice in python start point to *non inclusive* end
two_word_group = doc[0:3]
print(two_word_group.text)

four_word_group = doc[2:6]
print(four_word_group.text)


sectionheader("lexical attribute checking")
# Now we are going to update the doc for 
doc = nlp(
    "In 1990, more than 60% of people in East Asia were in extreme poverty. "
    "Now less than 4% are."
)

print(doc.text)


# Iterate through tokens in the doc
for token in doc:
    # Check if the token resembles a number
    if token.like_num:
        # Get the next token in the document
        next_token = doc[token.i + 1]
        # Check if the next token's text equals "%"
        if next_token.text == "%":
            print("Percentage found: ", token.text)



# to import a new pipeline, execute the following command in the pipeline "py -m spacy download en_core_web_sm"
sectionheader("Trained Pipelines")
nlp = spacy.load("en_core_web_md")
doc = nlp("she ate the pizza")
print(doc.text)

for token in doc: 
    # print the token and the part of speech tag, have spacy explain each part of speech tag
    print(token.text, token.pos_, f"({spacy.explain(token.pos_)})", token.dep_, f"({spacy.explain(token.dep_)})", token.head.text, )




sectionheader("Predicting Linguistic Annotations")
doc = nlp("It's official: Apple is the first U.S. public company to reach a $1 trillion market value")
print(doc.text)

for token in doc:
    # Get the token text, part-of-speech tag, and dependency label
    token_text = token.text
    token_pos = token.pos_
    token_dep = token.dep_
    print(f"{token_text:<12}{token_pos:<10} ({spacy.explain(token_pos):<12}) {token_dep:<10} ({spacy.explain(token_dep):<12}) ")


print()

# print each predicted entity and it's label
for ent in doc.ents:
    print(ent.text, ent.label_, f"({spacy.explain(ent.label_)})")



# 
sectionheader("Predicting Named Entities in Context")
doc = nlp("Upcoming iPhone X release date leaked as Apple reveals pre-orders")
print(doc.text)

for ent in doc.ents:
    print(ent.text, ent.label_)

iphone_x = doc[1:3]

print("Missing Entity: ", iphone_x.text)


sectionheader("Rule Based Matching with spacy.Matcher")

# Rule Based Matching
from spacy.matcher import Matcher
from helpermethods import printmatches

# initialize matcher with vocab from pipeline
matcher = Matcher(nlp.vocab)

# add the pattern to the matcher  
pattern = [{"TEXT": "iPhone"}, {"TEXT": "X"}]
matcher.add("IPHONE_PATTERN", [pattern])

printmatches(matcher(doc), doc)



print()
doc = nlp("I loved dogs buy now I love cats more.")

# add the pattern to the matcher
# LEMMA: the base form so {LEMMA: buy} could match buying or bought  
pattern = [
    {"LEMMA": "love", "POS": "VERB"},
    {"POS": "NOUN"}
]
matcher.add("LOVE_PATTERN", [pattern])
printmatches(matcher(doc), doc)


print()
doc = nlp("I bought a smartphone. Now I'm buying apps.")

pattern = [
    {"LEMMA": "buy"},
    {"POS": "DET", "OP": "?"}, # optional: match 0 or 1 times
    {"POS": "NOUN"}
]

matcher.add("BUYING_PATTERN", [pattern])
printmatches(matcher(doc), doc)

# potential "OP" values
# {"OP": "!"}	Negation: match 0 times
# {"OP": "?"}	Optional: match 0 or 1 times
# {"OP": "+"}	Match 1 or more times
# {"OP": "*"}	Match 0 or more times


print()
doc = nlp(
    "After making the iOS update you won't notice a radical system-wide "
    "redesign: nothing like the aesthetic upheaval we got with iOS 7. Most of "
    "iOS 11's furniture remains the same as in iOS 10. But you will discover "
    "some tweaks once you delve a little deeper."
)

# Write a pattern for full iOS versions ("iOS 7", "iOS 11", "iOS 10")
pattern = [
    {"TEXT": "iOS"}
    , {"IS_DIGIT": True}
]

matcher.add("IOS_PATTERN", [pattern])
printmatches(matcher(doc), doc)


print()
doc = nlp(
    "I downloaded Fortnite on my laptop and can't open the game at all. Help? "
    "so when I was downloading Minecraft, I got the Windows version where it "
    "is the '.zip' folder and I used the default program to unpack it... do "
    "I also need to download Winzip?"
)

# Write one pattern that only matches forms of “download” (tokens with the lemma “download”), followed by a token with the part-of-speech tag "PROPN" (proper noun).
pattern = [
    {"LEMMA": "download"}
    , {"POS": "PROPN"}
]

matcher.add("DOWNLOAD_THINGS_PATTERN", [pattern])
printmatches(matcher(doc), doc)


print()
doc = nlp(
    "Features of the app include a beautiful design, smart search, automatic "
    "labels and optional voice responses."
)

# Write one pattern that matches adjectives ("ADJ") followed by one or two "NOUN"s (one noun and one optional noun).
pattern = [
    {"POS": "ADJ"},
    {"POS": "NOUN"},
    {"POS": "NOUN", "OP": "?"}
]

matcher.add("ADJ_NOUN_PATTERN", [pattern])
printmatches(matcher(doc), doc)