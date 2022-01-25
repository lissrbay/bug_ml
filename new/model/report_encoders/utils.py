from re import split


def split_into_subtokens(name: str):
    return [word.lower() for word in split(r'(?=[A-Z])', name) if word]


def tokenize_frame(doc: str):
    return (word.lower() for token in doc.split(".") for word in split_into_subtokens(token))
