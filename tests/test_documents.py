from rag_language_tutor.documents import chunk_text_by_sentence, split_markdown_sections


def test_markdown_sections_are_split():
    sections = split_markdown_sections("# Title\nIntro\n\n## Rule\nUse this rule.")

    assert sections == [("Title", "Intro"), ("Rule", "Use this rule.")]


def test_chunking_preserves_text():
    chunks = chunk_text_by_sentence("One short sentence. Another useful sentence.", 30, 0)

    assert len(chunks) == 2
    assert chunks[0].startswith("One short")
