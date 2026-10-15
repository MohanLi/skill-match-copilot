from src.utils.pii import scrub
from src.utils.text import bio_to_spans, find_phrase_spans, phrases_to_bio, split_sentences, strip_html, tokenize


def test_tokenize_keeps_tech_tokens():
    assert tokenize("Experience with C++ , C# and node.js (3+ years).") == ["Experience", "with", "C++", ",", "C#", "and", "node.js", "(", "3+", "years", ")", "."]


def test_phrases_to_bio_and_back():
    toks = tokenize("Knowledge of Objective C and Java .")
    tags, unmatched = phrases_to_bio(toks, ["Objective C", "Java", "Kotlin"], "KNOW")
    assert unmatched == ["Kotlin"]
    assert [s[3] for s in bio_to_spans(toks, tags)] == ["Objective C", "Java"]


def test_find_phrase_spans_case_and_spacing():
    toks = ["Strong", "communication", "skills", "."]
    assert find_phrase_spans(toks, "communication   SKILLS") == [(1, 3)]


def test_strip_html_and_split():
    text = strip_html("<h2>Requirements</h2><ul><li>Python &amp; SQL</li><li>Docker. Kubernetes</li></ul><script>x</script>")
    sents = split_sentences(text)
    assert "Requirements" in sents[0] and "Python & SQL" in sents[1] and sents[2].startswith("Docker")
    assert "x" not in text


def test_pii_scrub():
    r = scrub("Call +380 67 123 45 67 or mail jane.doe@example.com, see https://linkedin.com/in/jane and @janedoe")
    assert "[PHONE]" in r.text and "[EMAIL]" in r.text and "[URL]" in r.text and "[HANDLE]" in r.text
    assert r.counts["email"] == 1 and r.counts["url"] == 1 and r.counts["phone"] == 1
    assert "2019-2022" in scrub("Worked 2019-2022 on 3 projects").text  # year ranges are not phones
