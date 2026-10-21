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


def test_strip_html_handles_escaped_html():
    # Greenhouse feeds deliver the posting body as escaped HTML
    raw = "&lt;div class=&quot;content&quot;&gt;&lt;p&gt;&lt;strong&gt;About us:&lt;/strong&gt; We build AI&amp;nbsp;tools.&lt;/p&gt;&lt;ul&gt;&lt;li&gt;Python&lt;/li&gt;&lt;li&gt;SQL&lt;/li&gt;&lt;/ul&gt;&lt;/div&gt;"
    text = strip_html(raw)
    assert "<" not in text and "&lt;" not in text
    assert text.splitlines()[0].startswith("About us: We build AI tools.")
    assert "Python" in text.splitlines() and "SQL" in text.splitlines()
    # plain text with entities and comparison signs is left intact
    assert strip_html("Strong communication &amp; planning; a < b and c > d") == "Strong communication & planning; a < b and c > d"


def test_champion_encoder_resolution(tmp_path, monkeypatch):
    from src.p2_normalize import normalizer as nm

    monkeypatch.setattr(nm, "models_cfg", lambda: {"p2_normalize": {"champion_encoder": "does/not/exist", "base_encoder": "models/base/all-MiniLM-L6-v2"}})
    monkeypatch.setattr(nm, "ROOT", tmp_path)
    (tmp_path / "models" / "base" / "all-MiniLM-L6-v2").mkdir(parents=True)
    assert nm.resolve_encoder() == tmp_path / "models" / "base" / "all-MiniLM-L6-v2"  # falls through to the base encoder
    (tmp_path / "models" / "p2_biencoder").mkdir()
    assert nm.resolve_encoder() == tmp_path / "models" / "p2_biencoder"  # in-house model beats the base encoder
    (tmp_path / "does" / "not" / "exist").mkdir(parents=True)
    assert nm.resolve_encoder() == tmp_path / "does" / "not" / "exist"  # configured champion wins when present
    assert nm.resolve_encoder(str(tmp_path / "models" / "base" / "all-MiniLM-L6-v2")) == tmp_path / "models" / "base" / "all-MiniLM-L6-v2"  # explicit path first
