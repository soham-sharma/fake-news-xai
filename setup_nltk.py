#!/usr/bin/env python3
"""Download required NLTK corpora for this project."""

import nltk

def main() -> None:
    corpora = ["punkt", "stopwords", "vader_lexicon"]
    for corpus in corpora:
        nltk.download(corpus)

if __name__ == "__main__":
    main()
