import re

class LexicalDebiaser:
    def __init__(self):
        self.known_leakage = ["reuters", "associated press"]
        
    def strip(self, text):
        return text.replace("reuters", "")

    def audit(self, df):
        # measuring how much of the fake class has the pattern vs true class
        pass
