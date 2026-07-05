# Section 9.6 — Agent Modeling with Neural Networks
### Presentation Guide for Seong Jin
**Slot length:** ~20 minutes (total session is 45 min, split across two presenters) | **Position:** second speaker, right after the ~2-min opening

---

## How to use this document

Slide-by-slide outline plus speaker notes, sized for a real 20-minute slot (roughly 10 slides, ~2 min average, with a couple of quicker ones and two slides — 5 and 6 — that deserve the most airtime). **[Book figure]** = an actual figure exists in the book; pull up the real one rather than relying on my description. **[If time allows]** = cut this first if you're running long; everything else is core. I've grounded section/subsection numbers and figure captions against the book's real table of contents, so those are accurate — the exact curve shapes in each figure are for you to read off your own copy.

---

## Learning objectives (what the room should take away)

1. Why explicitly modeling other agents is different from independent learning (9.3) or centralized critics/value decomposition (9.4–9.5).
2. How Deep JAL-AM scales tabular agent modeling (Ch. 6.3) up with neural networks.
3. The difference between predicting an agent's *next action* vs. learning a *representation* of its policy.
4. The bridge to Section 9.7: agent modeling is what you need when agents *aren't* similar enough to just share parameters.

---

## Slide 1 — Title (0.5 min)

**Content:** "Agent Modeling with Neural Networks" — Section 9.6, your name.

**Speaker notes:** One sentence into the topic, no throat-clearing: "This section is about what happens when, instead of treating other agents as noise, we build a model that predicts what they'll do."

---

## Slide 2 — Recap + Motivation (2 min)

**Content:**
- Chapter 9 so far: 9.3 Independent Learning → 9.4 Centralized Critics → 9.5 Value Decomposition
- Callback: Chapter 6.3 already did "agent modeling" (fictitious play, tabular JAL-AM) — 9.6 scales it up with neural nets
- The gap it fills: independent learning treats others as environment noise (non-stationarity); centralized critics/value decomposition fix credit assignment during training but don't require predicting anyone's behavior

**Speaker notes:** Merge the "where we are" and "why this matters" points into one tight beat — you don't have room for two separate slides here. Land the core distinction clearly: this section is about an agent explicitly predicting what others will do and using that prediction, which is orthogonal to both independent learning and CTDE-style credit assignment. It can be combined with either.

---

## Slide 3 — Tabular → Deep (1 min)

**Content:**
- Tabular JAL-AM (Ch. 6.3.2): keep frequency counts of others' actions per state — fine for small discrete spaces
- Breaks down with large/continuous observations — no table can cover it
- Fix: replace the table with a neural network mapping *observation → predicted action distribution*

**Speaker notes:** Quick bridge slide, don't linger — one sentence per bullet is enough. This is the same table-to-network move the book makes for value functions going from Chapter 6 to Chapters 8–9; agent modeling gets the same treatment.

---

## Slide 4 — 9.6.1: Deep JAL-AM — What It Is (2.5 min)

**Content:**
- Each agent *i* trains an **agent model network** per other agent *j*: π̂ⱼ(a | oⱼ)
- Trained via plain supervised learning — maximize likelihood of *j*'s actually observed actions (no RL involved in this part)
- Agent *i* computes its own Q-values as an **expectation over predicted actions of others** — an expected best response, not a guess

**Speaker notes:** This is your first "heart of the section" slide — take real time here even though the deck is short. The core mechanic: agent *i* doesn't just learn its own policy, it also learns a small model of every other agent, trained the boring reliable way (supervised, maximize likelihood of what *j* actually did). Once you have predicted probabilities for what everyone else will do, you can compute your own value estimates as an expectation over those predictions rather than being surprised by them — this is the deep-learning version of fictitious play's "best-respond to the empirical frequency of others' past actions."

---

## Slide 5 — Example + Limitations (3 min)

**[Book figure — Figure 9.20: level-based foraging, IDQN vs. Deep JAL-AM]**

**Content:**
- Pull up Figure 9.20 — walk through what the actual curves show
- Expected story: JAL-AM anticipates teammates' likely actions → faster/better coordination than IDQN, which experiences teammates as unpredictable noise
- **Limitations:** needs to *observe* others' actions (not always true); one model network per other agent → doesn't scale to large teams; only partially fixes non-stationarity — your model of *j* has to keep retraining as *j*'s policy keeps changing

**Speaker notes:** Combine the results slide and the limitations slide into one, since you don't have room for both separately. Show the real figure and set up the expected story before revealing it. Then pivot straight into limitations without a new slide — this is a good moment to invite some skepticism rather than let the technique sound like a silver bullet: it needs observable actions, it scales linearly (at best) with team size, and it converts "non-stationary environment" into "non-stationary supervised-learning target" rather than eliminating non-stationarity outright.

---

## Slide 6 — 9.6.2: Learning Representations of Agent Policies (3.5 min)

**[Book figure — Figure 9.21: encoder–decoder architecture]**

**Content:**
- Motivation: re-predicting raw actions every step is expensive and doesn't generalize — learn a reusable summary of "what kind of agent this is" instead
- **Encoder**: maps an agent's observed trajectory → a low-dimensional embedding
- **Decoder**: reconstructs/predicts that agent's actions from the embedding (training signal only — discarded after training)
- The embedding is concatenated into your **own** policy/value network's input

**Speaker notes:** This is your second and most important technical slide — give it the most time in the deck. The architecture is a classic encoder–decoder setup applied to *behavior*: the encoder compresses an agent's observed trajectory into a small embedding vector; the decoder's action-prediction task during training is what forces that embedding to actually mean something. Once trained, you throw away the decoder and just use the encoder as a standalone "agent typing" module — feed it a new agent's recent behavior, get back a compact summary you can condition your own policy on. Emphasize the generalization payoff versus Deep JAL-AM: this embedding can generalize to *new* agents with similar behavioral styles, not just re-predict the one agent you trained on.

---

## Slide 7 — Example Results (1.5 min)

**[Book figure — Figure 9.22: centralized A2C with/without representation-based agent models]**

**Content:** Pull up the real figure. Expected story: adding the learned representation as extra input to a centralized critic should improve sample efficiency or final performance vs. the same critic without it.

**Speaker notes:** Keep this brief — one beat: show the figure, state the expected direction of the result, move on. Worth one sentence noting this experiment layers agent modeling *on top of* a centralized critic (9.4), i.e., these techniques compose rather than compete.

---

## Slide 8 — Bridge to Section 9.7 (2 min)

**Content:**
- Learned agent embeddings show up again in a different context: **Selective Parameter Sharing** (Christianos, Papoudakis, Rahaman & Albrecht, 2021 — two of your book's own authors) clusters agents by learned embedding before deciding who shares network parameters
- One-line framing: *"9.6 is about handling agents that differ from you; 9.7 is about exploiting agents that are the same as you."*

**Speaker notes:** This is your handoff slide — make it land, since it's the connective tissue for the whole session. The same representation-learning idea from 9.6.2 shows up again as the tool that decides *whether* a group of agents is similar enough to benefit from the parameter-sharing techniques [your teammate] is about to cover. Say the one-line framing out loud as your literal transition sentence into the handoff.

---

## Slide 9 — Section 9.6 Summary + Handoff (1.5 min)

**Content:**
- Agent modeling = explicit, learned model of others instead of treating them as noise
- Two flavors: predict **actions** (Deep JAL-AM) vs. predict a general **representation** of policy
- Addresses non-stationarity/coordination — combinable with centralized critics and value decomposition
- "Over to [your name] for Section 9.7"

**Speaker notes:** Quick recap, no new content, then the literal handoff line. Don't summarize twice — say it once and sit down.

---

## Anticipated audience questions (prep these)

1. **"How is this different from a centralized critic?"** — A centralized critic gets privileged joint info only *during training* to fix credit assignment; the actor still doesn't predict others' behavior. Agent modeling gives an explicit, learned prediction of others' behavior, usable at execution time too if observability allows.
2. **"Doesn't this just move the non-stationarity problem?"** — Yes, fairly. It converts "non-stationary environment" into "non-stationary supervised-learning target for the agent model" — often easier to manage, not a full fix. Fine to say so plainly.
3. **"Could this work in a zero-sum competitive setting like 9.8?"** — In principle yes (recursive/theory-of-mind modeling connects to exploitability analysis), but 9.8 in the book relies on search + self-play rather than explicit opponent modeling — a fair connection to draw if it comes up, not something to claim the book does.

## If you have 2–3 spare minutes (optional, cut first)

- **Taxonomy aside:** Albrecht & Stone's 2018 survey (same Albrecht as your book) frames agent modeling as a family — policy reconstruction (what 9.6 does), type-based reasoning (Bayesian belief over discrete types, Ch. 6.3.3), and recursive/theory-of-mind reasoning ("I think that you think..."). Useful one-liner if someone asks "is this the only way to model other agents?"
