import re
text = "(reuters) - starting the news"
print(re.sub(r'\(reuters\)', '', text))
