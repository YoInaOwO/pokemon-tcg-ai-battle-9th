# Turning Lookahead into Features for PPO

**Subtitle:** Ninth place in the Simulation track with a 7.8M-parameter policy, without MCTS at play time.

Thank you to the organizers and Kaggle for hosting, and to all participants for the matches.

**TL;DR.** Both final submissions used the same decklist and model weights. Bounded engine lookahead supplies action consequences to a 7.8M-parameter policy that learns action sequencing and resource allocation. I trained it through behavioural cloning, value fine-tune and PPO against a pool based on the arena.

## 1. Deck: Ogerpon–Hydrapple

I chose a public list where Grass Energy helps draw cards, power attacks and increase damage.

| Package | Cards |
|---|---|
| Attackers | 4 Teal Mask Ogerpon ex; 2 each of Applin, Dipplin and Hydrapple ex |
| Evolution engine | 2 each of Chikorita, Bayleef and Meganium; 4 Forest of Vitality |
| Setup | 14 Basic Grass Energy; 4 Bug Catching Set; 4 Ultra Ball; 2 Poké Pad; 2 Dawn |
| Draw and recovery | 4 Lillie's Determination; 2 Meowth ex; 1 each of Fezandipiti ex, Night Stretcher and Lana's Aid |
| Interaction | 2 Boss's Orders; 1 each of Judge, Unfair Stamp and Tapu Bulu |

Teal Dance attaches Energy to Ogerpon and draws a card; Hydrapple's Ripening Charge attaches to any friendly Pokémon and heals it. Meganium's Wild Growth doubles the Energy each Basic Grass provides. Forest allows Grass Pokémon to evolve on the turn they enter play, except on the player's first turn.

The two main attacks depend on where the Energy is attached. Hydrapple's Syrup Storm counts Grass Energy across my board, while Ogerpon's Myriad Leaf Shower counts Energy on both Active Pokémon. Several Ogerpon provide more draws and attachments, but filling the bench can leave too little room for the evolution lines.

Bug Catching Set finds Energy and Grass Pokémon; Dawn finds the three evolution stages. Meowth can find a needed Supporter, but gives the opponent another two-Prize target. Recovery cards help rebuild after knockouts, while Boss's Orders lets me choose which Pokémon to knock out. Meganium and Tapu Bulu provide non-ex attacks against defensive abilities, though using them takes resources away from the main attackers.

## 2. Model architecture

![Policy network](model_redrawn.png)

*Figure 1. State encoding, option scoring and separate training heads. The opponent-hand input enters only the oracle critic.*

The observation becomes 64 tokens, which a five-layer Transformer with width 384 and six attention heads encodes.

Type and position embeddings distinguish zones and slots; padding masks exclude empty hand and revealed-card slots from attention.

Each decision supplies the visible state and all legal candidates. With global representation `h = X[0]`, a shared MLP scores `[o, h, o ⊙ h]`, combining option, state and their interaction.

The count head reads the global state and mean option representation. Its 24 classes cover counts 0–23; masks enforce the prompt's limits, and successive picks exclude previously selected options.

Two critics estimate returns during training. The oracle critic also receives an opponent-hand snapshot to reduce uncertainty in its value estimates and give PPO a more informative baseline. This private input never enters the policy. An auxiliary head predicts future Prize gains. At deployment, the policy runs in NumPy with engine features and without online MCTS.

## 3. Feature engineering

All 1,267 cards have 207 static features in the same format; fields that do not apply are zero.

| Card features | Width | Examples |
|---|---|---|
| Basic attributes | 57 | Card type, HP, Weakness, retreat cost, evolution stage and ex status |
| Ability and Trainer effects | 32 | Draw, search, attach Energy, heal, accelerate evolution; usage limits and effect magnitudes |
| Two attacks | 2 × 59 | Damage, Energy costs, coin flips and damage scaling with Energy or Prizes |

I extract effect fields from card text using rules and stored corrections. The model maps the card features to 64 values and adds a vector learned separately for each card. This preserves card identity even when effect features match. During training, ID dropout sometimes removes that separate vector, encouraging use of the shared features. These features cannot encode every rule.

Card representations combine with game information:

| State information | Width before embedding | Encoding |
|---|---|---|
| Board | 18 × (32 + 4 IDs) | HP, Energy, Tool count and first ID, cards beneath the Pokémon and arrival timing |
| Card zones | 30 hand; 8 revealed; 60 per bag | IDs, masks and counts for hand, revealed cards, discards and unseen cards (deck and unrevealed Prizes) |
| Global state | 40 + 21 | Turn order, resources, action usage and opponent belief |
| History | 26 + 26 IDs | Recent events and remembered public cards |
| Selection context | 68 + 2 IDs | Prompt and selection limits |
| Each candidate | 60 + 28 + 3 IDs | Base features, probe results; card, target and attack IDs |

Opponent belief estimates the opposing archetype. I match publicly revealed cards against replay decklists, weight by frequency and overlap, then aggregate into probabilities over 14 archetypes plus “other”. BC dropout sometimes replaces this distribution with “unknown”.

The probe lets the policy compare actions using their immediate consequences. For eligible main-phase, single-selection decisions, it executes up to 64 candidates in temporary engine states, follows forced continuations and branches on coin flips within fixed limits. It appends 28 features to each option, including damage, knockouts, Prize gains, hand/deck count changes and changes in legal attacks.

Consider attaching Energy to Hydrapple. Its board token combines 32 state values with four 64-value card representations: Pokémon, first Tool and first two Energy cards. The resulting 288 values are projected to 384. The attachment candidate combines 60 action features, 28 probe features and card, target and attack representations. The probe can flag an unlocked attack; cross-attention reads the whole state before the policy scores this attachment against other actions.

Hidden zones use a fixed completion to run these trials, so effects depending on hidden card identities remain approximate. The probe does not plan the opponent's response; option counts are suppressed after draws or searches.

## 4. Training

### Stage 1: behavioural cloning

I trained BC on replays from 3–12 August. The shared model learned to imitate recorded decisions from both players across all deck archetypes. I then initialized each archetype model from that checkpoint and fine-tuned the full network at a lower learning rate, using only decisions made while playing that archetype. Different exact decklists within an archetype contributed to the same model.

I weighted decisions from winners at 1.0 and losers at 0.3, giving more weight to recent games and stronger players.

### Stage 2: value fine-tune

Using the fine-tuned Ogerpon–Hydrapple BC model, I generated games and fitted both critics to their outcomes while keeping the policy frozen. PPO started from this value-tuned Ogerpon–Hydrapple checkpoint, with value estimates calibrated to states the policy visits.

### Stage 3: PPO

Pure self-play with my fixed deck would miss other archetypes' threats and Prize trades. I therefore built an opponent pool from the 12 August arena's 148 distinct lists.

| Opponent component | Share of games | Purpose |
|---|---|---|
| Top 40 exact 60-card lists | 75% | Sample in proportion to each list's arena frequency |
| Mutated lists | 10% | Sample variants uniformly to practise against altered card combinations |
| Four organizer-provided starter scripts | 10% | Cover simple but different play styles for the ladder's 10% random-opponent component |
| Self-mirrors | 5% | Practise the mirror matchup from both sides |

Each observed or mutated list uses its archetype's BC model, shared across that archetype's lists. Mirror games supply both players' trajectories, so they account for a larger share of training samples than of games.

For mutations, I start from each archetype's most common list and perform 5–10 replacement steps. Replacements come from cards observed within that archetype, capped at the largest count seen in any one list. The engine validates each resulting 60-card deck before inclusion.

Terminal rewards are +1 for wins, −1 for losses and 0 for draws. Clipping discourages abrupt policy changes, value regression improves return estimates, and the entropy bonus encourages exploration of alternative actions. The oracle critic supplies generalized advantage estimates. I used γ = 0.997, λ = 0.95, clipping 0.2 and learning rate 1e-4. Prize shaping and a penalty for drifting from the cloning policy anneal to zero over eight updates.

**Why such a large rollout?** I first increased the decisions collected per update from 131k to 524k and saw the training win-rate plateau rise. That result prompted a direct jump to 8.39 million, sixteen times the previous rollout.

With a small rollout, rare matchups contribute few games, so a lucky opening can have an outsized effect on an update. A larger rollout includes more games and both turn orders before each update. Mean absolute advantage fell from 0.35 initially to 0.16 near the peak training win rate. I think collecting more games helps separate small advantage estimates from game-to-game noise. The cost is fewer policy updates for the same number of decisions.

![Training progress and turn order](report_training_evidence.png)

*Figure 2. Left: arena-weighted training win rates using rolling windows of up to 300 games per fixed cloned opponent; scripts, mutants and mirrors are excluded. Right: first/second-player win rates during rollout 48; n combines both orders. Draws count as half a win.*

Using 8×RTX 4090, the final PPO run completed 57 updates in 56.3 hours. Its training win rate rose from 52.4% to 83.3% at update 48, after 403 million decisions; updates 45–57 stayed between 80.5% and 83.3%. Turn order remained important against Alakazam: 63.5% going first versus 50.6% second, compared with 83.9% versus 83.7% against Dragapult.

![Training win rates by archetype](training_archetypes.png)

*Figure 3. Training win rates against 12 BC archetypes, combining per-list rolling windows. All improved from the first recorded rollout to the last; Alakazam remained hardest.*

## 5. Results in real Kaggle battles

I analyzed 1,000 completed games from each final submission during 21–31 August: 1,144 wins, three draws and 853 losses overall. The overall win rate was 57.3%, counting draws as half a win. The two submissions achieved 56.8% and 57.8%, respectively.

Restricting the absolute pre-game rating gap to 200 leaves 1,800 games at **53.5%**, with an approximate 95% interval of **51.2–55.8%**. The other 200 games had a 91.5% win rate. This rating split helps interpret opponent strength; it does not identify the API's actual matchmaking mode.

![Win rate by opponent build](writeup_fig4_matchups.png)

*Figure 4. “Elo-matched” means rating gap ≤200. Groups with ≥15 games are shown; omitted games remain in the aggregate. “Mirror” includes all Hydrapple lists. Dots show win rates; grey bars show approximate Wilson 95% intervals. Repeated opponents can make the intervals optimistic.*

Among games with a rating gap ≤200, I won **53 of 74 games against the identical decklist: 71.6% [60.5–80.6%]**. With identical cards available to both sides, this result supports the policy's ability to use the deck effectively.

Although the win rates showed a gap between my BC models and Kaggle opponents, as imitation learning struggles to surpass those it imitates, these models still provided useful opponents for PPO training and a consistent benchmark for tracking the policy's progress.

Against Espeon–Sylveon, my win rate was 12.5% over 32 games. Sylveon's Safeguard blocks attack damage from my main ex attackers. Meganium and Tapu Bulu can bypass it, but Espeon's Psych Out can knock out either from full HP. This gives the opposing deck answers to both my main attackers and their backups.

## 6. What I tried that did not work

**MCTS.** I tried determinized MCTS with policy priors and critic leaf values, but found no useful improvement over the policy. With limited time per action, search must divide its work across sampled hidden states. Each tree also plans as if its sampled state were certain.

**A larger network.** More capacity did not help either. The network already receives detailed action consequences, so I chose to spend the budget on more training experience.

## 7. What I would improve

I managed my time poorly in this competition and ran out of time to train a new model for another deck. I would cap the first deck's training budget earlier, leaving enough time to train and evaluate a second deck before submission.

I would batch decisions from multiple games through a shared inference service to collect more training experience within the same budget.
