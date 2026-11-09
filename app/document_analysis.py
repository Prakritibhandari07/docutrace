
import re


STOP_WORDS = {
    "the", "is", "are", "was", "were", "a", "an",
    "and", "or", "of", "to", "in", "on", "for",
    "with", "from", "by", "as", "at", "this",
    "that", "these", "those", "it", "its", "be",
    "can", "may", "will", "which", "what", "how",
    "why", "where", "when", "who", "does", "do",
    "about", "into", "than", "then", "their",
    "there", "they", "them", "also"
}


def clean_words(text):
    words = re.findall(r"[A-Za-z0-9]+", text.lower())

    return [
        word
        for word in words
        if word not in STOP_WORDS and len(word) > 2
    ]


def extract_keywords(text, limit=10):
    words = clean_words(text)
    frequency = {}

    for word in words:
        frequency[word] = frequency.get(word, 0) + 1

    sorted_words = sorted(
        frequency.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return [
        word
        for word, count in sorted_words[:limit]
    ]


def find_keyword_pages(pages, keywords):
    keyword_pages = {}

    for keyword in keywords:
        matches = []

        for page in pages:
            text = page.get("text", "").lower()

            if keyword.lower() in text:
                matches.append(page.get("page", "Unknown"))

        keyword_pages[keyword] = matches

    return keyword_pages


def detect_headings(pages):
    headings = []

    for page in pages:
        text = page.get("text", "")
        lines = text.splitlines()

        for line in lines:
            line = line.strip()

            if not line:
                continue

            if len(line) > 100:
                continue

            words = line.split()

            if len(words) > 12:
                continue

            if (
                line.isupper()
                or line.endswith(":")
                or (
                    len(words) <= 8
                    and line[0].isupper()
                )
            ):
                headings.append({
                    "text": line,
                    "page": page.get("page", "Unknown")
                })

    return headings[:30]


def sentence_score(sentence, keywords):
    words = set(clean_words(sentence))
    overlap = words.intersection(set(keywords))

    return len(overlap)


def extractive_summary(text, keywords, max_sentences=3):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    scored = []

    for index, sentence in enumerate(sentences):
        sentence = sentence.strip()

        if len(sentence) < 30:
            continue

        score = sentence_score(sentence, keywords)

        scored.append((score, index, sentence))

    scored.sort(
        key=lambda item: (item[0], -item[1]),
        reverse=True
    )

    selected = sorted(
        scored[:max_sentences],
        key=lambda item: item[1]
    )

    return [
        item[2]
        for item in selected
    ]
