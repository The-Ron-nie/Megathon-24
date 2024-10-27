from textblob import TextBlob

# Sample text data
texts = [
    "I'm feeling great today! Excited about the project.",
    "I'm a bit overwhelmed and stressed out with work.",
    "I'm feeling neutral about the upcoming events.",
    "I've been feeling down lately, things aren't going well.",
    "I had a breakup recently which was toxic. My partner was abusive.",
    "Yaaaay! I got a promotion at work.Feeling ecstatic.",
   " I think my mental state is feeling hopeful.",
    "Honestly, I’m worried about health.",
    "It's been hard, I am extremely stressed.",
    "I feel feeling very low lately.",
    "I can’t stop happy and excited." , 
    "Every day I’m extremely stressed.",
"I am feeling very anxious these days.",
"Honestly, I’m happy and excited.",
"I feel happy and excited lately.",
"I think my mental state is extremely stressed.",
"It feels like I'm constantly feeling hopeful.",
"I feel extremely stressed lately.",
"I am can't sleep well, and it’s affecting me.",
"For a while now, I’ve been feeling much better.",
"Honestly, I’m happy and excited.",
"I don’t know why but I’ve been extremely stressed.",
"I’m trying, but I’m still feeling very anxious."
]

# Function to find polarity
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # Range is -1 (negative) to +1 (positive)
    if polarity > 0:
        return "Positive"
    elif polarity == 0:
        return "Neutral"
    else:
        return "Negative"

# Applying the function to each text
for text in texts:
    sentiment = analyze_sentiment(text)
    print(f"Text: {text}\nSentiment: {sentiment}\n")
