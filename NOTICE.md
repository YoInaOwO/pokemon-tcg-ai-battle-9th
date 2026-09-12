# Notice

This repository contains my agent code and final model weights for the Kaggle
*Pokémon Trading Card Game AI Battle Challenge*.

`models/hydra15038_model_weights.zip` contains `net.npz` extracted unchanged
from my final submission, plus a README with its checksum. It excludes the
competition engine, card database, decklist and opponent deck prior.

It deliberately does **not** contain, and must not be used to redistribute:

- the competition battle engine (`ptcgProgram`, the `cg` package and its
  compiled libraries), which ships under
  `LicenseRef-PTCG-ABC-Competition-Use-Only` and may not be reposted;
- the organisers' sample rule-based agents and starter notebooks;
- the card database (`EN/JP Card Data.csv`, card text dumps, the hand-audited
  card-skill schema derived from it);
- raw replay dumps from the competition arena and supporting datasets
  (per-day metadata and opponent pools with real card ids);
- complete submission bundles, which embed the engine.

Card and character names that appear in the code and in the write-up are
trademarks of Pokémon / Nintendo / Creatures / GAME FREAK. They are used here
only to describe what the agent does.

Training and game inference require the original competition dependencies,
including `cg`, obtained separately under their applicable license terms.
