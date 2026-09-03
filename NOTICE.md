# Notice

This repository contains **only my own code** for the Kaggle
*Pokémon Trading Card Game AI Battle Challenge*.

It deliberately does **not** contain, and must not be used to redistribute:

- the competition battle engine (`ptcgProgram`, the `cg` package and its
  compiled libraries), which ships under
  `LicenseRef-PTCG-ABC-Competition-Use-Only` and may not be reposted;
- the organisers' sample rule-based agents and starter notebooks;
- the card database (`EN/JP Card Data.csv`, card text dumps, the hand-audited
  card-skill schema derived from it);
- replay dumps from the competition arena, or anything derived from them
  (decklists, per-day metadata, opponent pools with real card ids);
- trained checkpoints and submission bundles, which embed the engine.

Card and character names that appear in the code and in the write-up are
trademarks of Pokémon / Nintendo / Creatures / GAME FREAK. They are used here
only to describe what the agent does.

Nothing here will run on its own: every entry point imports `cg`, which is
available only to competition participants.
