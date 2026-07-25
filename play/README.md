# Play the model

    python play/server.py          # then open http://localhost:8000

Plays the **pre-training** checkpoint (`ckpt/pretrain_best.pt`), the one trained to
predict which position comes next.

It does not choose moves. It scores candidate *next positions*: the server
generates every legal successor, encodes each with the project's own
`board_to_planes8`, and takes the model's highest-scoring one. The side panel
shows the full distribution, which is the interesting part — you can watch it be
confident in the opening and hopeless in an endgame.

- **Randomness** slider: 0 is greedy (same game every time); above 0 samples from
  the model's own probabilities.
- **What would it play?** scores the current position without moving.
- Inference is ~7 ms/move on CPU. No GPU needed.

Encoding runs through the training code rather than a JavaScript reimplementation.
A subtle mismatch there would just make the opponent quietly weaker, and nothing
would ever flag it.

## What to expect

Trained to imitate 1+0 bullet from ~1500-rated lichess players — not to play well.
It has no search and no evaluation function, only a learned sense of what a human
plays next. So it opens out of book (1.e4 at 52%, the Italian within six plies) and
then hangs pieces in the middlegame exactly like a human in time trouble.

Games longer than 160 plies exceed the trained position-embedding range; the server
uses the most recent 160, which is an approximation training never saw.
