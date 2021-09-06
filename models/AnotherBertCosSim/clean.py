import re
import pandas as pd

def cleanup(text):
    text = text.lower()
    text = re.sub(re.compile(r"<[^>]*>"), " ", text)
    text = re.sub(re.compile(r"^\[id\d*|.*\],*\s*"), "", text)
    text = re.sub(re.compile(r"(&quot;)|(&lt;)|(&gt;)|(&amp;)|(&apos;)"), " ", text)
    text = re.sub(re.compile(r"https?://(www\.)?[-a-zA-Z0-9@:%._+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_+.~#?&/=]*)"), " ", text)
    text = re.sub(re.compile(r"\[[^\[\]]+\|([^\[\]]+)\]"), r"\1", text)
    text = re.sub(re.compile(r"(&#\d+;)"), " ", text)
    text = re.sub(re.compile(r"[.,!?\-;:)(_#*=^/`@«»©…“•—<>\[\]\"'+%|&]"), " ", text)
    text = text.replace("  ", " ").replace("  ", " ").replace("  ", " ")
    return text

df = pd.read_csv("troll_data.csv", index_col=None).dropna()

df['question'] = df['question'].apply(cleanup)
df['answer'] = df['answer'].apply(cleanup)

df.to_csv("troll_data_cleaned.csv", index=None)