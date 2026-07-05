# Session Opening + Section 9.7 + Chapter 9 Big-Picture Summary
### Presentation Guide for You
**Total session: 45 minutes.** Rough budget: Opening (~2 min) → Seong Jin, Section 9.6 (~20 min) → You, Section 9.7 (~20 min) → You, Chapter 9 summary (~3 min). That's tight — treat the times below as ceilings, not targets, and know which slides to cut first if you're running long (marked **[If time allows]**).

---

## How to use this document

Same format as Seong Jin's. **[Book figure]** = pull up the real figure from your copy rather than relying on my description of it. **[Added context]** = supplementary material from outside the book, useful for depth/Q&A but cut first under time pressure. Section/subsection numbers and figure captions are grounded against the book's actual table of contents; the exact curve shapes in each figure are for you to read off your own copy.

---

# PART 1 — Opening (~2 minutes total)

## Slide O1 — Title + Roadmap (1.5–2 min)

**Content:**
- Course/seminar name, date, your name + Seong Jin's name
- Quick map: Chapter 9 so far (9.1–9.5) → today: **9.6 Agent Modeling** (Seong Jin) then **9.7 Homogeneous Agents** (you) → close with a Chapter 9 recap
- If relevant: flag that this is the last presentation of the semester on this book

**Speaker notes:** Keep this to under two minutes total — with only 45 minutes for the whole session, the opening is overhead, not content. One breath of welcome, one sentence on where we are in the book, straight to the handoff: "Over to Seong Jin for Section 9.6."

*(Seong Jin presents 9.6 for ~20 minutes here — see his document.)*

---

# PART 2 — Section 9.7: Environments with Homogeneous Agents (~20 minutes)

## Learning objectives

1. What makes agents "homogeneous" — strongly vs. weakly.
2. Parameter sharing: what it is, why it helps, when it backfires.
3. Experience sharing: how it differs, why it needs an off-policy correction.
4. That the two are complementary, not competing — and that this is the flip side of Seong Jin's section (similar agents vs. diverse agents).

## Slide 1 — Title + Transition (0.5 min)

**Content:** "Environments with Homogeneous Agents" — Section 9.7 (9.7.1 Parameter Sharing, 9.7.2 Experience Sharing)

**Speaker notes:** One-line pickup of Seong Jin's handoff: "9.6 was about teammates who behave differently. 9.7 flips that — what can we exploit when teammates are basically interchangeable?"

## Slide 2 — Motivation (1.5 min)

**Content:**
- Ch. 5.4.4: the joint action/state space can explode combinatorially as agent count grows
- Independent learning (9.3) trains N fully separate networks — wasteful if agents are similar
- Question: if agents *are* similar, can we avoid the N-times cost?

**Speaker notes:** One tight beat, not two. Land the point that independent learning throws away an obvious source of efficiency when agents are near-identical — any agent's experience is informative for the others too, precisely because they face near-identical decision problems.

## Slide 3 — Strongly vs. Weakly Homogeneous (3 min)

**[Book figure — Figure 9.23]**

**Content:**
- Homogeneous = same observation space + same action space
- **Strongly homogeneous:** fully interchangeable — optimal behavior pattern is symmetric across agents
- **Weakly homogeneous:** same interface, but optimal joint policy requires specialization (role, position, reward structure)
- Pull up Figure 9.23 for the book's concrete visual

**Speaker notes:** This is the single most important concept in the section — it determines whether the next few slides help or hurt. Use level-based foraging for continuity with Seong Jin: if only the agents actually involved in collecting an item get reward (recall Ch. 1's version), that reward structure alone pushes toward weak homogeneity even though the action/observation spaces are identical across agents.

## Slide 4 — Two Levers, and Parameter Sharing (4 min)

**Content:**
- Quick table: **Parameter sharing** (shares network weights — architecture level) vs. **Experience sharing** (shares training data — data level)
- Parameter sharing: one shared network φ instead of N separate ones; execution stays decentralized — each agent still only sees its own observation
- **Agent-ID trick:** concatenate a one-hot agent ID to the input so a shared network can still learn agent-specific behavior — needed because identical weights + identical inputs = identical outputs, which breaks weakly homogeneous settings
- Benefit: every gradient update effectively uses N agents' worth of data at once — a big sample-efficiency win, and cheap to implement

**Speaker notes:** This slide is doing the work of three from the original plan, so move briskly through the definition and benefit, but don't skip the agent-ID trick — it's a small detail that matters a lot in practice and it's a natural setup for the next slide's caveat. Land clearly: parameter sharing is fully compatible with decentralized execution — "centralizing" here just means training one network on everyone's data, no communication mechanism required.

## Slide 5 — Parameter Sharing: When It Backfires (2.5 min)

**[Added context]**

**Content:**
- Assumes real homogeneity — if agents' transition/reward functions genuinely differ, forced sharing can *hurt*, not help
- Can cause behavioral collapse: agents converge to identical (sometimes suboptimal) behavior
- **Selective Parameter Sharing (SePS)** — Christianos, Papoudakis, Rahaman & Albrecht (2021): learns agent embeddings (same spirit as Seong Jin's 9.6.2!) to cluster agents automatically, sharing only within clusters

**Speaker notes:** Explicitly call back to Seong Jin's section here — flag it as a deliberate callback so the room notices it. The representation-learning idea from 9.6.2 isn't just for opponent modeling — SePS uses almost the same encoder idea to decide *which* agents are similar enough to share parameters, rather than assuming all-or-nothing homogeneity.

## Slide 6 — Experience Sharing (3.5 min)

**Content:**
- Keep **separate** network parameters — no forced weight-tying
- Pool collected **transitions** across agents into a shared training signal — each agent still trains its own network, just on more data
- **Off-policy correction needed:** agent *j*'s transitions came from agent *j*'s policy, not agent *i*'s — using them naively biases agent *i* toward what *others* do
- Fix: weight borrowed experience by something like an importance-sampling ratio (how likely your *own* current policy would have taken that action, vs. how likely the original agent was to)

**Speaker notes:** This is your technical core for the section — most time here after the homogeneity slide. Make the off-policy point concrete: the moment you use *j*'s data to update *i*'s network, you've introduced an off-policy problem, and the importance-weighting fix down-weights experience that looks unlike what you yourself would do.

## Slide 7 — Concrete Example + Comparison (3 min)

**[Added context — flag as your own research, not verbatim book content]**
**[Book figure — Figure 9.24]**

**Content:**
- **SEAC** (Shared Experience Actor-Critic — Christianos, Schäfer & Albrecht, 2020): independent actor-critic (IA2C) plus an extra loss term per agent using other agents' experience, weighted by the correction above. Likely the concrete example behind 9.7.2 — verify against your copy before citing it as "the book's algorithm."
- Quick comparison table: parameter sharing (architecture, needs strong homogeneity, cheap) vs. experience sharing (data, tolerates weaker homogeneity, needs off-policy correction) — **can combine both**
- Pull up Figure 9.24 (level-based foraging learning curves) and read off what your copy actually shows

**Speaker notes:** Three things in one slide, so move at a good clip. State plainly that SEAC is your own addition, not something you're claiming verbatim from the book. On the comparison table, the one point to land verbally: these aren't competing techniques — they operate at different levels (architecture vs. data) and compose naturally. For the figure, set up the expected story (sharing → faster early learning, comparable or better final performance) before revealing the real curves.

## Slide 8 — Section 9.7 Summary (1.5 min)

**Content:**
- Homogeneity is a spectrum: strongly vs. weakly
- Parameter sharing: cheap, powerful, assumes real homogeneity (agent-ID trick preserves some specialization)
- Experience sharing: more flexible, needs an off-policy correction
- Same underlying resource (agent similarity), exploited at different levels

**Speaker notes:** Clean recap, no new content. Transition line into the closing: "That's 9.7 — and the last individual section for today, so let's zoom out to Chapter 9 as a whole."

---

# PART 3 — Chapter 9: The Big Picture (~3 minutes)

*Treat this as a fast, high-level close, not a third full presentation — one or two punchy slides, not a leisurely synthesis.*

## Slide 9 — Mapping Chapter 9 to Four Challenges (2 min)

**Content:**

| Section | Technique | Challenge addressed (from Ch. 5) |
|---|---|---|
| 9.3 Independent Learning | Treat others as environment | Baseline — *exposes* non-stationarity |
| 9.4 Policy Gradients | Centralized critics | Credit assignment, non-stationarity (training) |
| 9.5 Value Decomposition | VDN, QMIX | Credit assignment, at scale |
| 9.6 Agent Modeling | Deep JAL-AM, representations | Non-stationarity, coordination w/ diverse agents |
| 9.7 Homogeneous Agents | Parameter & experience sharing | Scaling, sample efficiency |
| 9.8 Self-Play | MCTS, AlphaZero | Equilibrium-finding, 2-player zero-sum |
| 9.9 Population-Based Training | PSRO, AlphaStar | Equilibrium-finding & robustness, general competitive |

**Speaker notes:** This one table is your entire big-picture argument — everything in Chapter 9 targets one of the four challenges from Chapter 5 (non-stationarity, equilibrium selection, credit assignment, scaling); none of these sections is "the" answer on its own. If 9.8/9.9 were covered in an earlier week, name that explicitly: "we already saw the competitive side of this chapter — today was the cooperative side."

## Slide 10 — Closing (1 min)

**Content:**
- One-line unifying thread: almost every section is a version of **centralized training, decentralized execution** — exploit extra info/compute during training, still act on local info at execution time
- Thank you / open floor

**Speaker notes:** Say the CTDE line once, don't unpack it in depth — there isn't time. Thank the room and Seong Jin, and since this is the last presentation of the semester on this book, explicitly invite questions about the material as a whole, not just today's two sections.

---

## Anticipated audience questions (prep these)

1. **"If parameter sharing is so cheap, why bother with value decomposition (VDN/QMIX)?"** — Different problems: parameter sharing buys sample efficiency when agents are similar; it doesn't by itself solve difficult coordination/credit-assignment problems. You can combine both.
2. **"Doesn't experience sharing risk copying a struggling teammate's bad habits?"** — That's exactly what the importance-weighting correction guards against, though imperfectly — fair to admit this is a real limitation, especially early in training when all agents' policies are still bad and hard to tell apart.
3. **"Which technique would you use for football/RoboCup-style agents?"** — Good bridge if it comes up: same-team players are weakly homogeneous (same interface, but real positional specialization matters), which points toward experience sharing or clustered/selective parameter sharing rather than full parameter sharing — forcing one identical network on every position would wash out the specialization the strategy depends on.

## If you're running ahead of schedule (optional, add back in order)

1. Agent-ID trick gets its own dedicated slide instead of being folded into Slide 4.
2. A short "what the book left out" note (communication learning, evolutionary game theory — flagged as out of scope in the book's own preface) before the closing slide.
3. The CTDE thread (Slide 10) expanded into two slides: one for the pattern itself, one connecting Chapter 6's tabular algorithms to Chapter 9's deep versions as a running theme across the whole semester.
