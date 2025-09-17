from src.preprocess import chunk_text, clean_text

def test_chunk_and_clean():
    text = "This is a test. " * 100
    cleaned = clean_text(text)
    chunks = chunk_text(cleaned, chunk_words=20, overlap=5)
    assert len(chunks) >= 4
    assert all(len(c.split()) <= 20 for c in chunks)
