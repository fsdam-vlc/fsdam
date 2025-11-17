import copy
from collections import defaultdict, Counter
import numpy as np
import math
from nltk.tokenize import RegexpTokenizer
from scipy.stats.mstats import gmean


# =========================
# Helper Penalty Functions
# =========================
def compute_penalty_by_length(candidate_len, reference_len, alpha=1.0):
    """Gaussian penalty for sentence length difference."""
    delta = abs(reference_len - candidate_len)
    return np.e ** (-(delta ** 2) / (float(alpha) * float(reference_len) ** 2))


def compute_penalty_by_repetition(candidate_sent, reference_sent, penalty_func=lambda x: np.exp(-x)):
    """Penalty for word repetition difference."""
    tokenizer = RegexpTokenizer(r'\w+')
    tokens_candidate = tokenizer.tokenize(candidate_sent)
    tokens_reference = tokenizer.tokenize(reference_sent)

    word_freq_candidate = Counter(tokens_candidate)
    word_freq_reference = Counter(tokens_reference)

    scores = []
    for word, freq in word_freq_candidate.items():
        if word in word_freq_reference:
            diff = abs(word_freq_reference[word] - freq)
            scores.append(penalty_func(diff))
        else:
            scores.append(penalty_func(freq - 1))

    if len(scores) == 0:
        return 1.0
    return gmean(np.array(scores))


# =========================
# Core TF-IDF Processing
# =========================
def precook(s, n=4, out=False):
    """Tokenize a sentence and count n-grams."""
    if isinstance(s, list):
        s = s[0]
    if not isinstance(s, str):
        s = str(s)

    words = s.split()
    counts = defaultdict(int)
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i:i + k])
            counts[ngram] += 1
    return counts


def cook_refs(refs, n=4):
    """Convert reference captions into n-gram dictionaries."""
    return [precook(ref, n) for ref in refs]


def cook_test(test, n=4):
    """Convert a test caption into n-gram dictionary."""
    if isinstance(test, list):
        test = test[0]
    return precook(test, n, True)


# =========================
# CIDEr-R Scorer
# =========================
class CiderRScorer(object):
    """CIDEr-R scorer with repetition and length penalties."""

    def copy(self):
        new = CiderRScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        return new

    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        self.n = n
        self.sigma = sigma
        self.test = []
        self.refs = []
        self.crefs = []
        self.ctest = []
        self.document_frequency = defaultdict(float)
        self.cook_append(test, refs)
        self.ref_len = None

    def cook_append(self, test, refs):
        """Append a new hypothesis-reference pair."""
        if refs is not None:
            self.refs.append(refs)
            self.test.append(test)
            self.crefs.append(cook_refs(refs))
            if test is not None:
                self.ctest.append(cook_test(test))
            else:
                self.ctest.append(None)

    def size(self):
        assert len(self.crefs) == len(self.ctest), \
            f"refs/test mismatch! {len(self.crefs)} <> {len(self.ctest)}"
        return len(self.crefs)

    def __iadd__(self, other):
        """Combine another scorer or tuple."""
        if isinstance(other, tuple):
            self.cook_append(other[0], other[1])
        else:
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)
        return self

    def compute_doc_freq(self):
        """Compute document frequency of n-grams in the reference corpus."""
        for refs in self.crefs:
            for ngram in set([ngram for ref in refs for (ngram, _) in ref.items()]):
                self.document_frequency[ngram] += 1

    def compute_cider(self):
        """Compute the CIDEr-R score."""
        def counts2vec(cnts):
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for (ngram, term_freq) in cnts.items():
                df = np.log(max(1.0, self.document_frequency[ngram]))
                n = len(ngram) - 1
                vec[n][ngram] = float(term_freq) * (self.ref_len - df)
                norm[n] += pow(vec[n][ngram], 2)
                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref,
                sent_hyp, sent_ref, coefficient_rep=0.8, coefficient_len=0.2):
            """Cosine similarity with penalties."""
            val = np.array([0.0 for _ in range(self.n)])
            delta = float(length_hyp - length_ref)

            for n in range(self.n):
                for (ngram, _) in vec_hyp[n].items():
                    val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]
                if norm_hyp[n] != 0 and norm_ref[n] != 0:
                    val[n] /= (norm_hyp[n] * norm_ref[n])

                rep_penalty = compute_penalty_by_repetition(sent_hyp, sent_ref,
                                                            penalty_func=lambda x: 1 / (1 + x))
                len_penalty = compute_penalty_by_length(length_hyp + 1, length_ref + 1)
                val[n] *= (rep_penalty ** coefficient_rep) * (len_penalty ** coefficient_len)
            return val

        self.ref_len = np.log(float(len(self.crefs)))
        scores = []

        for i, (test, refs) in enumerate(zip(self.ctest, self.crefs)):
            if test is None or len(test) == 0:
                scores.append(0.0)
                continue

            vec, norm, length = counts2vec(test)
            score = np.array([0.0 for _ in range(self.n)])
            for j, ref in enumerate(refs):
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref,
                             sent_hyp=str(self.test[i][0] if isinstance(self.test[i], list) else self.test[i]),
                             sent_ref=str(self.refs[i][j][0] if isinstance(self.refs[i][j], list) else self.refs[i][j]))

            score_avg = np.mean(score) / len(refs) * 10.0
            scores.append(score_avg)

        return scores

    def compute_score(self, option=None, verbose=0):
        """Compute final corpus-level CIDEr-R score."""
        self.compute_doc_freq()
        assert len(self.ctest) >= max(self.document_frequency.values())
        score = self.compute_cider()
        return np.mean(np.array(score)), np.array(score)


# =========================
# CIDEr-R Wrapper
# =========================
class CiderR:
    """Main interface for CIDEr-R metric."""

    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        self._n = n
        self._sigma = sigma
        self.cider_scorer = CiderRScorer(n=self._n, sigma=self._sigma)

    def append(self, gts, res):
        self.cider_scorer += (res, gts)

    def compute_score(self):
        return self.cider_scorer.compute_score()

    def method(self):
        return "CIDEr-R"

