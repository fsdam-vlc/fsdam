# meteor.py
import os
import subprocess
import threading
from pathlib import Path

# Paths
METEOR_JAR = str(Path(__file__).parent / "meteor-1.5.jar")
PARAPHRASE_FILE = str(Path(__file__).parent / "data/paraphrase-en.gz")

class Meteor:
    def __init__(self, sep2w=False):
        self.env = os.environ.copy()
        self.env["LC_ALL"] = "en_US.UTF-8"

        # Use system Java (Java 11+)
        self.meteor_cmd = [
            "/usr/bin/java",
            "-jar",
            "-Xmx2G",
            METEOR_JAR,
            "-",
            "-",
            "-stdio",
            "-l",
            "en",
            "-norm"
        ]

        # Start main METEOR subprocess
        self.meteor_p = subprocess.Popen(
            self.meteor_cmd,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=self.env,
            universal_newlines=True,
            bufsize=1,
        )

        self.sep2w = sep2w
        self.lock = threading.Lock()

        if sep2w:
            self.meteor_p_wt = subprocess.Popen(
                self.meteor_cmd,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.env,
                universal_newlines=True,
                bufsize=1,
            )
            self.meteor_p_wy = subprocess.Popen(
                self.meteor_cmd,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.env,
                universal_newlines=True,
                bufsize=1,
            )
            self.eval_line_wt = "EVAL"
            self.eval_line_wy = "EVAL"
            self.scores_wt = []
            self.scores_wy = []
            self.count_wt = 0
            self.count_wy = 0

        self.scores = []
        self.eval_line = "EVAL"
        self.count = 0

    def append(self, gts, res, mode=None):
        """Add a new example to evaluate."""
        self.lock.acquire()
        if self.sep2w:
            if mode == "what":
                stat = self._stat(res, gts, mode="what")
                self.eval_line_wt += f" ||| {stat}"
                self.count_wt += 1
            elif mode == "why":
                stat = self._stat(res, gts, mode="why")
                self.eval_line_wy += f" ||| {stat}"
                self.count_wy += 1
        else:
            stat = self._stat(res, gts)
            self.eval_line += f" ||| {stat}"
            self.count += 1
        self.lock.release()

    def compute_score(self):
        """Compute METEOR score."""
        self.lock.acquire()
        self.meteor_p.stdin.write(self.eval_line + "\n")

        if self.sep2w:
            self.meteor_p_wt.stdin.write(self.eval_line_wt + "\n")
            self.meteor_p_wy.stdin.write(self.eval_line_wy + "\n")

        for _ in range(self.count):
            score = float(self.meteor_p.stdout.readline().strip())
            self.scores.append(score)

        final_score = float(self.meteor_p.stdout.readline().strip())

        if self.sep2w:
            for _ in range(self.count_wt):
                score_wt = float(self.meteor_p_wt.stdout.readline().strip())
                self.scores_wt.append(score_wt)
            for _ in range(self.count_wy):
                score_wy = float(self.meteor_p_wy.stdout.readline().strip())
                self.scores_wy.append(score_wy)
            final_score_wt = float(self.meteor_p_wt.stdout.readline().strip())
            final_score_wy = float(self.meteor_p_wy.stdout.readline().strip())

        self.lock.release()

        if self.sep2w:
            return (
                final_score,
                self.scores,
                final_score_wt,
                self.scores_wt,
                final_score_wy,
                self.scores_wy,
            )

        return final_score, self.scores

    def _stat(self, hypothesis_str, reference_list, mode=None):
        """Format input line for METEOR process."""
        hypothesis_str = hypothesis_str.replace("\n", " ").replace("|||", "").replace("  ", " ")
        reference_list[0] = reference_list[0].replace("\n", " ").replace("|||", "").replace("  ", " ")
        score_line = " ||| ".join(("SCORE", " ||| ".join(reference_list), hypothesis_str))

        if self.sep2w:
            if mode == "what":
                self.meteor_p_wt.stdin.write(score_line + "\n")
                return self.meteor_p_wt.stdout.readline().strip()
            elif mode == "why":
                self.meteor_p_wy.stdin.write(score_line + "\n")
                return self.meteor_p_wy.stdout.readline().strip()
            else:
                raise ValueError('mode must be "what" or "why"')
        else:
            self.meteor_p.stdin.write(score_line + "\n")
            return self.meteor_p.stdout.readline().strip()

    def compute_method(self):
        return "METEOR"

    def __del__(self):
        """Clean up subprocesses."""
        try:
            self.lock.acquire()
            if hasattr(self, "meteor_p") and self.meteor_p:
                self.meteor_p.stdin.close()
                self.meteor_p.kill()
                self.meteor_p.wait()
            if self.sep2w:
                for p in [self.meteor_p_wt, self.meteor_p_wy]:
                    p.stdin.close()
                    p.kill()
                    p.wait()
        finally:
            self.lock.release()
