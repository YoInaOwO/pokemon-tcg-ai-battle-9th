# Turning Lookahead into Features for PPO

**Subtitle:** Ninth place in the Simulation track with a 7.8M-parameter policy, without MCTS at play time.

Thank you to the organizers and Kaggle for hosting, and to all participants for the matches.

**TL;DR.** Both final submissions used the same decklist and model weights. A 7.8M-parameter network scores legal actions from the board state and short engine probes. I trained it through behavioural cloning, value fine-tune and PPO against a pool based on the arena. The final PPO run used 8.39 million decisions per update and took 56.3 hours. The agent finished ninth.

## 1. Deck: Ogerpon–Hydrapple

I chose a public list where Grass Energy helps draw cards, power attacks and increase damage. Getting these effects in the right order is central to playing the deck.

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

The observation becomes 64 tokens covering global state, selection context, board slots, hand, revealed cards, discard piles, public history, memory and my unseen cards (deck and unrevealed Prizes). A five-layer Transformer with width 384 and six attention heads encodes them. Card representations combine learned ID embeddings with projected static attributes and structured Ability/attack descriptions. ID dropout encourages the model to use those descriptions when card identities are unfamiliar.

Type and position embeddings distinguish zones and slots; padding masks exclude empty hand and revealed-card slots from attention.

Each legal option is represented by its features and the card, target and attack it refers to. Cross-attention lets it read the state: an option targeting a bench slot can read that Pokémon's HP and Energy. The first encoded token gives the global representation, `h = X[0]`. An MLP scores `[o, h, o ⊙ h]`, combining option, state and their interaction. A separate head predicts the selection count.

The same scoring network evaluates each candidate. It can therefore handle prompts with different numbers of options and reuse what it learns across actions.

The count head reads the global state and mean option representation. Its 24 classes cover counts 0–23; masks enforce the prompt's limits, and successive picks exclude previously selected options.

Two critics estimate returns during training; the oracle critic additionally receives an opponent-hand snapshot. An auxiliary head predicts future Prize gains. Deployment runs the policy in NumPy, with engine features and without online MCTS.

## 3. Feature engineering

I encode resources, public history and action consequences explicitly.

| Block | Width | Information and purpose |
|---|---|---|
| Global and opponent belief | 40 + 21 | Turn order, resource counts, once-per-turn usage; beliefs over 14 archetypes plus “other” supply matchup context. |
| Selection context | 68 + 2 IDs | Prompt type, selection limits and remaining requirements distinguish attacks, searches and forced choices. |
| Board | 18 × (32 + 4 IDs) | HP, damage, Energy, tools, evolution and arrival timing describe targets and development constraints. |
| Card zones | IDs, masks, counts | 30 hand slots, 8 revealed-card slots, discard bags and my unseen-card pool describe accessible and recoverable resources. |
| Log and memory | 26 + 26 IDs | Recent events, cumulative play/attachment counts, four attacks per side and up to 12 known opponent cards preserve information across decisions. |
| Base option | 60 + IDs | Action type, location, card, target, attack, counts and target attributes identify what each candidate does. |
| Engine lookahead | 28 | Damage, knockouts, Prize/resource changes and newly legal attacks expose immediate consequences. |

Opponent belief estimates deck archetypes from publicly revealed opponent cards. I match these against exact replay decklists, weight matches by frequency and card overlap, then sum weights by archetype and normalize. This distribution enters the global state. Cloning dropout sometimes replaces it with “unknown” to reduce reliance on deck identification.

For eligible main-phase, single-selection decisions, the probe evaluates up to 64 candidates, following forced continuations and coin-flip branches within a bounded expansion.

HP is scaled by 400. Selection counts use logarithms so that large forced selections remain distinguishable. Target attributes and action effects together let the same card receive different scores in different positions.

For example, attaching Energy may make Hydrapple's attack legal. The probe exposes that change; the policy still decides whether to attack now or use another Ability first to reach a knockout threshold. The engine supplies local consequences while the network learns sequencing and resource tradeoffs.

The probe fills hidden zones with a fixed completion rather than the opponent's actual hand. Some option-count features are suppressed after draws or searches so they do not depend on that completion's shuffle. The resulting features describe short-term consequences; they do not predict the opponent's full response.

## 4. Training

### Stage 1: behavioural cloning

I built the replay corpus from 136k episodes collected between 14 July and 12 August. I first trained a shared model across deck families, then fine-tuned it for each archetype. Sharing skills such as attaching and evolving helps decks with fewer demonstrations.

I weight decisions from winners at 1.0 and losers at 0.3, with ten-day recency decay and extra weight for stronger players.

### Stage 2: value fine-tune

Before PPO, I generated games with the starting policy and fitted both critics to their outcomes, keeping the policy frozen. This calibrates value estimates to states the agent actually visits. PPO uses the difference between returns and predicted values to guide its updates, so calibrating the critics first gives it a more useful baseline.

### Stage 3: PPO

I was training one deck, so pure self-play would focus on mirror matches and miss other decks' threats and Prize trades. I built an opponent pool from the 12 August arena's 148 distinct lists, keeping mirrors as one component.

| Opponent component | Purpose |
|---|---|
| Top 40 exact 60-card lists | Sample in proportion to each list's arena frequency |
| Mutated lists | Sample variants uniformly to practise against altered card combinations |
| Four organizer-provided starter scripts | Cover simple but different play styles for the ladder's 10% random-opponent component |
| Self-mirrors | Practise the mirror matchup from both sides |

The pool builder assigns these components 75%, 10%, 10% and 5% of games. Each observed or mutated list uses its archetype's BC model, shared across that archetype's lists. Mirror games supply both players' trajectories, so they account for a larger share of training samples than of games.

For mutations, I start from each archetype's most common list and perform 5–10 replacement steps. Replacements come from cards observed within that archetype, capped at the largest count seen in any one list. The engine validates each resulting 60-card deck before inclusion.

Clipping discourages abrupt policy changes, value regression improves return estimates, and the entropy bonus encourages exploration of alternative actions. The oracle critic supplies generalized advantage estimates. I used γ = 0.997, λ = 0.95, clipping 0.2 and learning rate 1e-4. Prize shaping and a penalty for drifting from the cloning policy anneal to zero over eight updates, leaving the final objective focused on winning.

**Why such a large rollout?** I first increased the decisions collected per update from 131k to 524k and saw the training win-rate plateau rise. That result prompted a direct jump to 8.39 million, sixteen times the previous rollout. Training win rate rose slowly at first, then reached a higher plateau.

With a small rollout, rare matchups contribute few games, so a lucky opening can have an outsized effect on an update. A larger rollout includes more games and both turn orders before each update. Mean absolute advantage fell from 0.35 initially to 0.16 near the peak training win rate. My interpretation is that broader sampling helps distinguish these smaller signals from game-to-game variation. The cost is fewer policy updates for the same number of decisions, which helps explain the slower initial progress.

![Training progress and turn order](report_training_evidence.png)

*Figure 2. Left: arena-weighted training win rates using rolling windows of up to 300 games per fixed cloned opponent; scripts, mutants and mirrors are excluded. Right: first/second-player win rates during rollout 48; n combines both orders. Draws count half.*

With eight RTX 4090 GPUs, the run completed 57 updates in 56.3 hours. Its training win rate rose from 52.4% to 83.3% at update 48, after 403 million decisions; updates 45–57 stayed between 80.5% and 83.3%. Turn order remained important against Alakazam: 63.5% going first versus 50.6% second, compared with 83.9% versus 83.7% against Dragapult.

## 5. Results in real Kaggle battles

I analyzed 1,000 completed games from each final submission during 21–31 August: 1,144 wins, three draws and 853 losses overall. Counting draws as half a win gives 57.3%; the submissions separately scored 56.8% and 57.8%.

Restricting the absolute pre-game rating gap to 200 leaves 1,800 games at **53.5%**, with an approximate 95% interval of **51.2–55.8%**. The remaining 200 games scored 91.5%. This rating split helps interpret opponent strength; it does not identify the API's actual matchmaking mode.

![Win rate by opponent build](writeup_fig4_matchups.png)

*Figure 3. “Elo-matched” means rating gap ≤200. Groups with ≥15 games are shown; omitted games remain in the aggregate. “Mirror” includes all Hydrapple lists. Grey bars show approximate Wilson 95% intervals; repeated opponents can make them optimistic.*

Within this group, I won **53 of 74 games against the identical decklist: 71.6% [60.5–80.6%]**. These games help assess how well the policy plays with the cards held fixed, though opponent strength still varies.

I scored 55.4% against Dragapult's Jamming Tower build and 42.9% against Risky Ruins. Against Espeon–Sylveon, I scored just 12.5% over 32 games: having non-ex attackers did not make that plan reliable. My 45.7% against Alakazam over 138 games showed how much harder real competitors were than the cloned training opponents.

## 6. What I tried that did not work

**MCTS.** I tried determinized MCTS with policy priors and critic leaf values, but found no useful improvement over the policy. Under a short action budget, search must divide its work across sampled hidden states. Each tree also plans as if its sampled state were certain. I retained bounded feature probes for submission.

**A larger network.** More capacity did not help either. The network already receives detailed action consequences, so I chose to spend the budget on more training experience.

## 7. What I would improve

I would use a shared GPU service to batch decisions from multiple games. Faster inference would let me collect more training experience within the same budget.

I would set aside time for a second deck earlier and give the first deck a firm training budget. That would leave time to train and evaluate an alternative before submission.
