from de_pipeline.stages.chunk import chunk_text, n_tokens, split_sentences

TEXT = " ".join(f"Sentence number {i} has exactly seven words." for i in range(40))


def test_split_sentences():
    assert split_sentences("Hello there. How are you? Fine!\n\nNew para.") == [
        "Hello there.", "How are you?", "Fine!", "New para."]


def test_chunks_respect_budget_and_overlap():
    chunks = chunk_text(TEXT, size=50, overlap=15, min_tokens=10)
    assert len(chunks) > 1
    assert all(n_tokens(c) <= 50 for c in chunks)
    for a, b in zip(chunks, chunks[1:]):           # consecutive chunks share sentences
        assert split_sentences(a)[-1] in split_sentences(b)
    # every sentence is covered
    covered = {s for c in chunks for s in split_sentences(c)}
    assert covered == set(split_sentences(TEXT))


def test_long_sentence_hard_split_and_small_doc():
    long = " ".join(["word"] * 120) + "."
    assert all(n_tokens(c) <= 50 for c in chunk_text(long, 50, 10, 5))
    assert chunk_text("Just one short sentence.", 50, 10, 5) == ["Just one short sentence."]
