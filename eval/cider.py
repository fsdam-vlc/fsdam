import copy
from collections import defaultdict
import numpy as np
import math


def precook(s, n=4, out=False):
    """
    Convert a sentence into a frequency map of n-grams.
    """
    # Ensure s is a string, not list
    if isinstance(s, list):
        s = s[0]
    words = s.split()
    counts = defaultdict(int)
    for k in range(1, n + 1):
        for i in range(len(words) - k + 1):
            ngram = tuple(words[i:i + k])
            counts[ngram] += 1
    return counts


def cook_refs(refs, n=4):
    """Process all reference captions."""
    return [precook(ref, n) for ref in refs]


def cook_test(test, n=4):
    """Process hypothesis caption."""
    if isinstance(test, list):
        test = test[0]
    return precook(test, n, True)


class CiderScorer(object):
    """CIDEr scorer."""

    def copy(self):
        new = CiderScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        return new

    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []
        self.document_frequency = defaultdict(float)
        self.cook_append(test, refs)
        self.ref_len = None

    def cook_append(self, test, refs):
        """Add a new sample pair."""
        if refs is not None:
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
        """Accumulate more samples."""
        if isinstance(other, tuple):
            self.cook_append(other[0], other[1])
        else:
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)
        return self

    def compute_doc_freq(self):
        """Compute document frequency for reference corpus."""
        for refs in self.crefs:
            for ngram in set([ngram for ref in refs for (ngram, count) in ref.items()]):
                self.document_frequency[ngram] += 1

    def compute_cider(self):
        """Compute CIDEr score."""
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

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            val = np.array([0.0 for _ in range(self.n)])
            delta = float(length_hyp - length_ref)
            for n in range(self.n):
                for (ngram, _) in vec_hyp[n].items():
                    val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]
                if norm_hyp[n] != 0 and norm_ref[n] != 0:
                    val[n] /= (norm_hyp[n] * norm_ref[n])
                val[n] *= np.e ** (-(delta ** 2) / (2 * self.sigma ** 2))
            return val

        self.ref_len = np.log(float(len(self.crefs)))
        scores = []
        for test, refs in zip(self.ctest, self.crefs):
            vec, norm, length = counts2vec(test)
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            score_avg = np.mean(score) / len(refs) * 10.0
            scores.append(score_avg)
        return scores

    def compute_score(self, option=None, verbose=0):
        self.compute_doc_freq()
        assert len(self.ctest) >= max(self.document_frequency.values())
        score = self.compute_cider()
        return np.mean(np.array(score)), np.array(score)


class Cider:
    """Main CIDEr interface."""

    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        self._n = n
        self._sigma = sigma
        self.cider_scorer = CiderScorer(n=self._n, sigma=self._sigma)

    def append(self, gts, res):
        self.cider_scorer += (res, gts)

    def compute_score(self):
        return self.cider_scorer.compute_score()

    def method(self):
        return "CIDEr"

