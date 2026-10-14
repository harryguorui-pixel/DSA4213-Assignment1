
import math

from src.ngram import TrigramLanguageModel


def test_ngram():

    # A tiny artificial corpus.
    # 2 represents EOS.
    tokens = [
        3, 4, 5, 2,
        3, 4, 5, 2,
        3, 4, 6, 2
    ]

    model = TrigramLanguageModel(
        vocab_size=7,
        k=0.1,
        pad_id=0,
        eos_id=2
    )

    model.fit(tokens)

    # Context (3, 4) occurs three times.
    assert model.bigram_counts[(3, 4)] == 3

    # Trigram (3, 4, 5) occurs twice.
    assert model.trigram_counts[(3, 4, 5)] == 2

    # Trigram (3, 4, 6) occurs once.
    assert model.trigram_counts[(3, 4, 6)] == 1

    # Probability distribution must sum to 1.
    total_prob = sum(
        model.probability((3, 4), token)
        for token in range(1, 7)
    )

    assert math.isclose(
        total_prob,
        1.0,
        rel_tol=1e-9
    )

    # Seen trigram should have higher probability.
    assert (
        model.probability((3, 4), 5)
        >
        model.probability((3, 4), 6)
    )

    results = model.evaluate(tokens)

    assert math.isfinite(results["cross_entropy"])
    assert math.isfinite(results["perplexity"])
    assert results["evaluated_tokens"] == 6

    print("All n-gram tests passed!")


if __name__ == "__main__":
    test_ngram()
