
import math
import random
from collections import Counter, defaultdict


class TrigramLanguageModel:
    """
    Trigram language model with Add-k smoothing.

    Predicts the next token using the previous two tokens.
    """

    def __init__(self, vocab_size, k=0.1, pad_id=0, eos_id=2):
        self.vocab_size = vocab_size
        self.k = k
        self.pad_id = pad_id
        self.eos_id = eos_id

        # Exclude PAD from possible next-token predictions.
        self.predictable_vocab_size = vocab_size - 1

        self.trigram_counts = Counter()
        self.bigram_counts = Counter()

        self.context_next_counts = defaultdict(Counter)

        self.num_training_tokens = 0

    def fit(self, token_ids):
        """
        Count trigrams from the training corpus.

        EOS is used as a boundary: the model never predicts
        a token across an EOS boundary.
        """

        tokens = [
            int(token)
            for token in token_ids
        ]

        # Reset counts if fit is called again.
        self.trigram_counts.clear()
        self.bigram_counts.clear()
        self.context_next_counts.clear()

        self.num_training_tokens = len(tokens)

        context = []

        for token in tokens:

            if token == self.pad_id:
                continue

            if len(context) == 2:
                u, v = context

                self.trigram_counts[(u, v, token)] += 1
                self.bigram_counts[(u, v)] += 1
                self.context_next_counts[(u, v)][token] += 1

            if token == self.eos_id:
                context = []
            else:
                context.append(token)
                context = context[-2:]

        print("Trigram training completed.")
        print("Training tokens:", self.num_training_tokens)
        print("Unique trigrams:", len(self.trigram_counts))
        print("Unique contexts:", len(self.bigram_counts))

    def probability(self, context, next_token):
        """
        Calculate P(next_token | previous two tokens).
        """

        if next_token == self.pad_id:
            return 0.0

        context = tuple(context[-2:])

        count_context = self.bigram_counts[context]

        count_trigram = self.trigram_counts[
            (*context, next_token)
        ]

        numerator = count_trigram + self.k

        denominator = (
            count_context
            + self.k * self.predictable_vocab_size
        )

        return numerator / denominator

    def evaluate(self, token_ids):
        """
        Calculate average next-token cross-entropy
        and perplexity.

        Only targets with two preceding tokens within
        the same text segment are evaluated.
        """

        total_negative_log_likelihood = 0.0
        total_predictions = 0

        context = []

        for token in token_ids:

            token = int(token)

            if token == self.pad_id:
                continue

            if len(context) == 2:

                prob = self.probability(
                    context,
                    token
                )

                total_negative_log_likelihood -= math.log(prob)

                total_predictions += 1

            if token == self.eos_id:
                context = []
            else:
                context.append(token)
                context = context[-2:]

        if total_predictions == 0:
            raise ValueError("No valid tokens for evaluation.")

        avg_loss = (
            total_negative_log_likelihood
            / total_predictions
        )

        perplexity = math.exp(avg_loss)

        return {
            "cross_entropy": avg_loss,
            "perplexity": perplexity,
            "evaluated_tokens": total_predictions,
        }

    def generate(
        self,
        prompt_ids,
        max_new_tokens=50,
        strategy="greedy",
        temperature=1.0,
        seed=42
    ):
        """
        Generate tokens from a prompt.

        Supported strategies:
            greedy
            sample
        """

        if temperature <= 0:
            raise ValueError("Temperature must be positive.")

        rng = random.Random(seed)

        generated = [
            int(token)
            for token in prompt_ids
        ]

        context = []

        for token in generated:
            if token == self.eos_id:
                context = []
            else:
                context.append(token)
                context = context[-2:]

        for _ in range(max_new_tokens):

            if len(context) < 2:
                # No complete trigram context available.
                # Stop rather than inventing a context.
                break

            candidate_ids = range(1, self.vocab_size)

            if strategy == "greedy":

                next_token = max(
                    candidate_ids,
                    key=lambda token: self.probability(
                        context,
                        token
                    )
                )

            elif strategy == "sample":

                probabilities = [
                    self.probability(context, token)
                    for token in candidate_ids
                ]

                # Apply temperature scaling.
                weights = [
                    p ** (1.0 / temperature)
                    for p in probabilities
                ]

                next_token = rng.choices(
                    range(1, self.vocab_size),
                    weights=weights,
                    k=1
                )[0]

            else:
                raise ValueError(
                    "strategy must be 'greedy' or 'sample'"
                )

            generated.append(next_token)

            if next_token == self.eos_id:
                break

            context.append(next_token)
            context = context[-2:]

        return generated

    def model_statistics(self):
        """
        Return count-based model statistics.
        """

        return {
            "unique_trigrams": len(self.trigram_counts),
            "unique_contexts": len(self.bigram_counts),
            "training_tokens": self.num_training_tokens,
            "smoothing_k": self.k,
        }
