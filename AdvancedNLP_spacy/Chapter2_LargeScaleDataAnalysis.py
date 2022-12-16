# Chapter 2: Large-scale data analysis with spaCy
# In this chapter, you'll use your new skills to extract specific information from large volumes of text. 
# You'll learn how to make the most of spaCy's data structures, and how to effectively combine statistical and rule-based approaches for text analysis.
from helpermethods import sectionheader
import spacy

sectionheader("Data Structures (1)")
sectionheader("Vocab, Lexemes and StringStore", 2)

# Vocab: stores data shared across multiple documents
# String Store: It's a lookup table that works in both directions. You can look up a string and get its hash, and look up a hash to get its string value. Internally, spaCy only communicates in hash IDs.
# Hash IDs can't be reversed, though. If a word is not in the vocabulary, there's no way to get its string. That's why we always need to pass around the shared vocab.
nlp = spacy.blank("en")

nlp.vocab.strings.add("coffee")
coffee_hash = nlp.vocab.strings["coffee"]
coffee_string = nlp.vocab.strings[coffee_hash]

doc = nlp("I love coffee")
print("hash value: ", nlp.vocab.strings["coffee"])
print("string value: ", nlp.vocab.strings[3197928453018144401])
print("hash value from doc:", doc.vocab.strings["coffee"])


sectionheader("Coffee Lexeme", 3)
# Lexeme: an entry in the vocabulary
lexeme = nlp.vocab["coffee"]

# Print the lexical attributes
for property, value in vars(lexeme).items():
    print(f"{property}: {value}")



sectionheader("Strings to Hashes", 2)
# Look up the string “cat” in nlp.vocab.strings to get the hash.
# Look up the hash to get back the string.

doc = nlp("I have a cat")

# Look up the hash for the word cat
cat_hash = nlp.vocab.strings["cat"]
print("Cat Hash: ", cat_hash)