# Session Opening + Section 9.7 + Chapter 9 Big-Picture Summary
### Presentation Guide for You
**Your slots:** (1) short opening (~3–5 min), (2) Section 9.7 (~40 min, ceiling 45 min), (3) closing Chapter 9 summary (~10–12 min)
**Position in session:** you open → hand off to Seong Jin (9.6) → he hands back to you → you present 9.7 → you close with the chapter summary

---

## How to use this document

Same format as Seong Jin's: slide-by-slide outline plus full speaker notes. **[Book figure]** means an actual figure exists in the book — pull up the real one rather than relying on my description of it. **[Added context]** flags supplementary material from outside the book (useful for depth and Q&A, cut first if you're short on time). I've grounded the section numbers, subsection numbers, and figure captions against the book's actual table of contents, so those are accurate; the specific curve shapes/numbers in each figure are for you to read off your own copy since I don't have the book's exact experimental results memorized.

---

# PART 1 — Opening the Session (~3–5 minutes)

## Slide O1 — Title Slide

**Slide content:**
- Course/seminar name, date
- "Chapter 9: Multi-Agent Deep Reinforcement Learning — Sections 9.6 & 9.7"
- Your name + Seong Jin's name

**Speaker notes:**
Welcome the room, quick logistics (timing, that this is the last presentation of the semester if that's worth flagging up front so people know to save bigger-picture questions for the end).

## Slide O2 — Where We Are in the Book

**Slide content:**
- Quick map of Chapter 9 so far: 9.1 Training/Execution Modes → 9.2 Notation → 9.3 Independent Learning → 9.4 Multi-Agent Policy Gradients (Centralized Critics) → 9.5 Value Decomposition
- Today: **9.6 Agent Modeling** (Seong Jin) → **9.7 Homogeneous Agents / Parameter & Experience Sharing** (you)
- Coming up (not us, but worth naming for context): 9.8 Self-Play in Zero-Sum Games, 9.9 Population-Based Training

**Speaker notes:**
Give the room a quick "you are here" map before diving in — this is genuinely useful for a chapter this dense, and it's a nice callback if earlier sessions this semester already covered 9.1–9.5. If your class has *already* covered 9.8/9.9 in an earlier week (worth confirming beforehand), phrase this slide as "and to close out the chapter, we'll also fit those into the big picture at the end" rather than "coming up." Keep this to under a minute — it's context, not content.

## Slide O3 — Handoff to Seong Jin

**Slide content:**
- "Section 9.6: Agent Modeling with Neural Networks — Seong Jin"

**Speaker notes:**
"With that context set, I'll hand over to Seong Jin to walk us through Section 9.6." Then sit down / step back — no need to say more.

*(Seong Jin presents 9.6 for ~40 minutes here — see his separate document.)*

---

# PART 2 — Section 9.7: Environments with Homogeneous Agents (~40 minutes)

## Learning objectives for this section

By the end, the room should be able to:
1. Define what makes a set of agents "homogeneous," and distinguish strongly vs. weakly homogeneous.
2. Explain parameter sharing: what it is, why it helps, and when it can backfire.
3. Explain experience sharing: how it differs from parameter sharing, and why it needs an off-policy correction.
4. Articulate the trade-off between the two techniques, and know that they can be combined.
5. See how this section connects back to Seong Jin's — homogeneity is the *opposite end of the spectrum* from the diverse-agent case that agent modeling handles.

## Slide 1 — Title Slide

**Slide content:**
- "Environments with Homogeneous Agents" — Section 9.7
- Subsections: 9.7.1 Parameter Sharing, 9.7.2 Experience Sharing

**Speaker notes:**
Quick transition line picking up Seong Jin's handoff cue: "Section 9.6 was about what to do when your teammates behave differently from you and from each other. Section 9.7 flips that around — what can you exploit when your teammates are basically *interchangeable* with you?"

## Slide 2 — Recall: The Scaling Challenge from Chapter 5

**Slide content:**
- Chapter 5.4.4: number of possible action combinations can grow exponentially with the number of agents
- Independent learning (9.3) trains **N completely separate networks** — wasteful if agents are similar to begin with
- Question for 9.7: if agents *are* similar, can we avoid paying that N-times cost?

**Speaker notes:**
This is the motivating problem, so make sure it lands before moving to definitions. Independent learning, which was introduced early in Chapter 9 as the simplest possible baseline, assigns each agent its own completely independent network. If you have 10 nearly identical robots doing nearly identical jobs, training 10 separate networks from scratch is throwing away an obvious source of efficiency: any experience one robot gathers about "what works" is, in principle, informative for the other 9 as well, precisely because they're facing near-identical decision problems. Section 9.7 is about formalizing "how similar do agents need to be" and then giving you two concrete techniques to exploit that similarity.

## Slide 3 — What Does "Homogeneous" Actually Mean?

**Slide content:**
- Same **observation space** and same **action space** across agents
- Ideally: **interchangeable** — swapping which agent is in which role wouldn't change the optimal joint behavior
- This is a spectrum, not a binary property

**Speaker notes:**
Be precise here, since "homogeneous" gets used loosely elsewhere in ML. The formal requirement is about the *interface* — same observation and action spaces — but the deeper, more useful notion is interchangeability: could you swap two agents' roles and get the same quality of outcome? A group of identical delivery robots in an empty warehouse is close to perfectly interchangeable. A group of identical-looking robots that have nonetheless specialized — one always guards the north exit, one always restocks a particular aisle — share the same interface but are no longer strictly interchangeable in practice.

## Slide 4 — Strongly vs. Weakly Homogeneous Agents

**[Book figure — Figure 9.23]**

**Slide content:**
- **Strongly homogeneous:** agents are fully interchangeable — same dynamics, same optimal behavior pattern in symmetric situations
- **Weakly homogeneous:** same observation/action space, but the optimal joint policy requires agents to specialize or behave differently (e.g., due to position, role, or reward structure)
- Pull up the real Figure 9.23 here to show the book's concrete visual examples

**Speaker notes:**
This distinction is the single most important concept in this section, because it's exactly what determines whether the techniques on the next few slides will help you or hurt you. Use the level-based foraging example to make this concrete for continuity with Seong Jin's slides: if every agent gets the same shared reward regardless of who actually collected an item, the setting is closer to strongly homogeneous — any agent doing "the right generic thing" is equally good. But recall from Chapter 1's version of level-based foraging that only the agents that were actually involved in collecting an item get the positive reward — this reward structure alone can push agents toward weak homogeneity, since it creates pressure for agents to differentiate (e.g., rush toward different items) even though their action/observation spaces are identical.

## Slide 5 — Two Levers to Exploit Homogeneity

**Slide content:**

| | What's shared | Level |
|---|---|---|
| **Parameter Sharing (9.7.1)** | Network weights | Architecture |
| **Experience Sharing (9.7.2)** | Training data (transitions) | Data |

**Speaker notes:**
Give the room this map before diving into either one individually — it prevents people from conflating the two, which is an extremely common confusion in the MARL literature itself. Parameter sharing is a decision about your network architecture: do all agents literally use the same weights? Experience sharing is a decision about your training data pipeline: regardless of whether weights are shared, do agents pool the transitions they individually collect into a common training signal? You can do either alone, both together, or neither (plain independent learning). We'll take them in that order.

## Slide 6 — Parameter Sharing: The Basic Idea

**Slide content:**
- Instead of N separate policy networks φ₁, φ₂, ..., φₙ → use **one shared network** φ
- Execution is still **decentralized**: each agent feeds in only its own local observation
- The shared network just happens to be the same function for everyone

**Speaker notes:**
Stress the "decentralized execution" point, since it's easy to accidentally imply that parameter sharing means centralization — it doesn't. At execution time, agent *i* still only sees its own observation and produces its own action; the only thing that's shared is which numbers make up the network doing that computation. This means parameter sharing is fully compatible with the CTDE (centralized training, decentralized execution) framework this whole chapter is built around, and it's actually one of the cheapest ways to implement CTDE, since "centralizing" here just means "train one network using everyone's data" rather than requiring any explicit communication mechanism between agents.

## Slide 7 — The Agent-ID Trick

**Slide content:**
- Problem: if the network is *identical* for everyone and inputs are identical, outputs will be identical — bad for weakly homogeneous settings
- Fix: concatenate a unique **agent ID** (e.g., one-hot vector) to each agent's observation before feeding it into the shared network
- Same weights, but the network can still learn agent-specific behavior conditioned on ID

**Speaker notes:**
This is a small implementation detail that matters a lot in practice, so don't skip it. Without some way to break symmetry, a literally-shared network fed literally-identical inputs will produce literally-identical outputs — which is fine for strongly homogeneous settings, but a real problem for weakly homogeneous ones where you actually need agents to specialize (recall the level-based foraging reward structure from a couple of slides ago). Appending an agent ID gives the single shared network enough information to learn *conditionally* different behavior per agent, without needing separate weights — you still get all the sample-efficiency benefits of sharing while retaining the ability to specialize.

## Slide 8 — Parameter Sharing: Benefits

**Slide content:**
- Sample efficiency: every gradient update uses experience from **all N agents at once** — effectively multiplies your batch size by N
- Fewer total parameters to learn → faster, often more stable training
- Simpler to implement than most credit-assignment machinery (VDN/QMIX/COMA)

**Speaker notes:**
The sample-efficiency point is the headline benefit and worth dwelling on: if you have 10 agents and they all contribute their experience to training the *same* network, then in a very real sense each gradient step has effectively seen 10 times as much data as a single independent agent would have collected on its own in the same number of environment steps. This is a big part of why parameter sharing is often the *first* thing people try in cooperative MARL before reaching for more complex machinery like value decomposition or centralized critics — it's cheap, simple to implement (barely more than a for-loop change in most codebases, as the book's own companion code demonstrates), and often gives a solid chunk of the total possible improvement over naive independent learning.

## Slide 9 — Parameter Sharing: Risks and When It Backfires

**[Added context]**

**Slide content:**
- Assumes homogeneity — if agents have genuinely different transition/reward functions, forcing shared weights can *hurt* performance
- Can cause "behavioral collapse": agents converge to identical (sometimes suboptimal) behavior even where specialization would be better
- **Selective Parameter Sharing (SePS)** — Christianos, Papoudakis, Rahaman & Albrecht (2021): learns agent embeddings (similar spirit to Seong Jin's 9.6.2!) and automatically clusters agents, sharing parameters only *within* clusters

**Speaker notes:**
This slide is a great place to explicitly call back to Seong Jin's section — flag it as a deliberate callback so the room notices the connective tissue between the two halves of today's presentation. Full parameter sharing is a strong assumption, and empirically it can genuinely hurt when it doesn't hold: research studying this directly found that when different agents' transition or reward functions are meaningfully distinct, the representations that a fully-shared network is forced to learn become harder to form well, and full sharing stops being effective — sometimes performing *worse* than fully independent learning. The proposed fix, Selective Parameter Sharing, learns an embedding of each agent's behavior — using almost the same encoder idea Seong Jin just described in Section 9.6.2 — and then runs unsupervised clustering on those embeddings to automatically decide which agents are similar enough to share a network, sharing separately per cluster rather than assuming all-or-nothing homogeneity. It's a nice illustration of how "learn a representation of an agent" (9.6) and "decide who should share parameters" (9.7) aren't actually separate problems — they're two applications of the same underlying idea.

## Slide 10 — Experience Sharing: The Basic Idea

**Slide content:**
- Keep **separate** network parameters per agent — no forced weight-tying
- But: pool the collected **transitions** (observation, action, reward, next-observation) across agents into a shared training signal
- Each agent still trains its *own* network, just on a bigger, shared pool of data

**Speaker notes:**
Contrast this directly with parameter sharing on the previous slides: this time, nothing changes architecturally — agent *i* still has its own network, agent *j* still has its own network, potentially with different weights. What changes is the *data* each network gets trained on. Instead of agent *i*'s network only ever seeing transitions that agent *i* itself experienced, it also gets to learn from transitions that its teammates experienced. This is a more flexible way to exploit homogeneity than parameter sharing, because it doesn't force agents to compute the same function — it just lets them learn faster by pooling data, while still allowing each agent's policy to end up different from the others if that's what the task actually needs.

## Slide 11 — Experience Sharing: The Off-Policy Correction Problem

**Slide content:**
- Agent *j*'s transitions were generated by agent *j*'s policy, not agent *i*'s
- Using them naively in agent *i*'s update is like doing **off-policy learning** without correcting for it
- Fix: weight agent *j*'s experience using something like an importance-sampling ratio — how likely agent *i*'s *own* current policy would have been to take that same action

**Speaker notes:**
This is the technical crux of experience sharing, so slow down here. The moment you use agent *j*'s experience to update agent *i*'s network, you've quietly introduced an off-policy learning problem: the data was generated by one policy (agent *j*'s), but you're using it to improve a different policy (agent *i*'s). If you ignore this and just dump all agents' experience into one big pool for everyone, you risk biasing each agent's learning toward whatever the *other* agents happen to be doing, rather than what's actually good for that specific agent. The standard fix, borrowed from off-policy actor-critic methods, is to weight each borrowed transition by something like the ratio of "how likely would MY current policy have been to take this action" divided by "how likely was the ORIGINAL agent's policy to take this action" — down-weighting experience that looks very unlike what you yourself would do, and up-weighting experience that's more representative.

## Slide 12 — A Concrete Algorithm: Shared Experience Actor-Critic (SEAC)

**[Added context]**

**Slide content:**
- Christianos, Schäfer & Albrecht (2020) — again, two of your book's own authors
- Builds on independent actor-critic (IA2C): each agent keeps its own actor and critic
- Adds an extra loss term per agent using **other agents' experience**, weighted by the importance-sampling-style ratio from the previous slide

**Speaker notes:**
I'd flag this explicitly as your own added research rather than something pulled verbatim from the book, since it's likely the concrete algorithm underlying this section but you should double check this against your actual copy of 9.7.2 before presenting it as "the book's example." Structurally, SEAC is a very clean illustration of "experience sharing without parameter sharing": each agent runs ordinary independent actor-critic, and on top of its normal loss (computed from its own experience) it adds a second loss term computed from every other agent's recently collected experience, correcting for the fact that this borrowed experience came from a different policy using the importance-weighting idea from the last slide. The appeal is that it's a fairly small, modular addition on top of a simple baseline (IA2C) — you don't need to change the network architecture at all, just the training loop.

## Slide 13 — Parameter Sharing vs. Experience Sharing: Comparison

**Slide content:**

| | Parameter Sharing | Experience Sharing |
|---|---|---|
| Level of sharing | Architecture (weights) | Data (transitions) |
| Assumes agents are... | Strongly homogeneous (ideally) | Can tolerate weaker homogeneity |
| Preserves per-agent specialization? | Only via agent-ID trick | Yes, naturally |
| Implementation cost | Very low | Moderate (needs off-policy correction) |
| Can combine both? | **Yes** — nothing stops you |  |

**Speaker notes:**
Use this table to consolidate the whole section before moving to results. The key takeaway to land verbally, since it's easy to walk away thinking these are competing techniques: they're not mutually exclusive. You could share parameters *and* additionally share experience across a cluster of agents that share those parameters — the two levers operate at different levels (architecture vs. data) and compose naturally. Which one (or both) you reach for in practice depends on how strongly homogeneous your agents actually are, which loops back to the strongly-vs-weakly distinction from Slide 4.

## Slide 14 — Example Results: Level-Based Foraging

**[Book figure — Figure 9.24]**

**Slide content:**
- Environment: level-based foraging (same recurring environment as Seong Jin's examples — nice continuity)
- Pull up the actual Figure 9.24 learning curves here
- Compare independent learning against the sharing variant(s) shown in your copy of the book

**Speaker notes:**
As with Seong Jin's example figures, present the real curves from your copy rather than reconstructing specific numbers — I don't have the book's exact experimental values in front of me, so treat this slide as a placeholder for "insert your book's actual figure and read off what it shows." The story to set up before revealing the curve: if the sharing technique(s) shown are working as intended, you'd expect faster early learning (steeper initial climb — the sample-efficiency benefit) and possibly a higher or comparable final performance plateau relative to plain independent learning, since the agents in level-based foraging are close to homogeneous by construction. If the real curves show something more complicated (e.g., a technique that learns faster early but plateaus lower), that's an even better talking point — it's a nice natural segue into asking the room *why* that might happen, tying back to the strongly/weakly homogeneous distinction.

## Slide 15 — Where This Fits in the Bigger MARL Toolbox

**Slide content:**
- Centralized critics (9.4) / value decomposition (9.5): solve **credit assignment**
- Agent modeling (9.6): handles **non-stationarity and diversity** between agents
- Parameter & experience sharing (9.7): solves **sample efficiency and scaling** to many agents
- These are complementary axes, not competing solutions to the same problem

**Speaker notes:**
This slide previews your closing summary, so keep it tight, but it's worth planting the idea now that each section of Chapter 9 is best understood as targeting a *different* one of the core challenges from Chapter 5 (non-stationarity, equilibrium selection, credit assignment, scaling), rather than all of them competing to be "the" solution to multi-agent RL. You'll return to and expand this exact framing in the closing summary.

## Slide 16 — Section 9.7 Summary

**Slide content:**
- Homogeneity is a spectrum: strongly vs. weakly homogeneous agents
- Parameter sharing: share weights, cheap, powerful, but assumes real homogeneity (agent-ID trick helps preserve some specialization)
- Experience sharing: share data instead, more flexible, needs an off-policy correction (e.g., SEAC)
- Both exploit the *same* underlying resource — agent similarity — at different levels (architecture vs. data)

**Speaker notes:**
Clean recap, no new content. Transition line into the closing summary: "That wraps up 9.7, and with that we've actually covered the last of the individual sections for today — so let's zoom out and look at Chapter 9 as a whole."

---

# PART 3 — Chapter 9: The Big Picture (~10–12 minutes)

*(This is the closing section — pitch this as more reflective/synthesis-oriented than the detailed technical sections above. Slower pace, more discussion-friendly.)*

## Slide 17 — Title Slide

**Slide content:**
- "Chapter 9 — The Big Picture"
- Subtitle: "Multi-Agent Deep Reinforcement Learning, section by section"

**Speaker notes:**
Frame this explicitly as the "why does any of this hang together" section — since this is the last presentation of the semester on this book, it's worth spending a little extra time here making the whole chapter feel like a coherent story rather than a list of ten separate algorithms.

## Slide 18 — Recall: The Four Core Challenges (Chapter 5)

**Slide content:**
1. **Non-stationarity** — other agents are learning too, so the environment appears to keep changing
2. **Equilibrium selection** — multiple valid equilibria may exist; which one do agents converge to?
3. **Multi-agent credit assignment** — whose action caused the shared outcome?
4. **Scaling to many agents** — the joint action/state space can explode combinatorially

**Speaker notes:**
These four challenges, introduced back in Chapter 5, are the organizing thread for the whole "big picture" argument you're about to make — every technique in Chapter 9 exists to make progress on at least one of these four. If your class covered Chapter 5 a while ago, take an extra moment here to briefly remind people what each challenge actually means, since the payoff of the next slide depends on the room remembering these.

## Slide 19 — Mapping Chapter 9 to the Four Challenges

**Slide content:**

| Section | Main technique | Challenge(s) addressed |
|---|---|---|
| 9.3 Independent Learning | Treat others as environment | Baseline — *exposes* non-stationarity rather than solving it |
| 9.4 Multi-Agent Policy Gradients | Centralized critics | Credit assignment, non-stationarity (during training) |
| 9.5 Value Decomposition | VDN, QMIX (IGM property) | Credit assignment, at greater scale than 9.4 |
| 9.6 Agent Modeling | Deep JAL-AM, representation learning | Non-stationarity, coordination with diverse agents |
| 9.7 Homogeneous Agents | Parameter & experience sharing | Scaling to many agents, sample efficiency |
| 9.8 Self-Play (zero-sum) | MCTS, AlphaZero | Equilibrium-finding in competitive 2-player games |
| 9.9 Population-Based Training | PSRO, AlphaStar | Equilibrium-finding & robustness in general competitive settings |

**Speaker notes:**
This is the centerpiece slide of your entire closing summary — spend real time on it and be prepared to walk through each row. The point to drive home: none of these sections is "the" answer to multi-agent deep RL; each one is best understood as a specialized tool aimed at one (or two) of the four challenges from Chapter 5. This is also a good moment to explicitly connect back to material from earlier in the semester if your seminar has already covered self-play/MCTS or PSRO/meta-games in a previous week — you can say something like "we actually already looked at how 9.8 and 9.9 use search and population-based training to handle equilibrium selection in competitive settings; today's sections showed you the cooperative side of the same chapter."

## Slide 20 — The Common Thread: CTDE

**Slide content:**
- **C**entralized **T**raining, **D**ecentralized **E**xecution shows up almost everywhere in this chapter:
  - 9.4's centralized critics: privileged joint info only during training
  - 9.5's value decomposition: a centralized joint value factored for decentralized action selection
  - 9.7's parameter sharing: "centralizing" by training one shared network on everyone's data
  - 9.9's population-based training: massive centralized self-play, producing independently executable policies
- 9.6 and 9.8 are the two outliers worth naming explicitly: agent modeling *can* work in a fully decentralized setting too, and self-play's "centralization" is really about compute/infrastructure rather than information-sharing per se

**Speaker notes:**
This slide ties the whole chapter together under one paradigm, which is a satisfying way to close things out. CTDE was introduced all the way back in Section 9.1 as one of three training/execution modes, and if you look carefully, almost every subsequent section in the chapter is really just a different concrete instantiation of that same idea: exploit whatever extra information or computation you can afford during training, as long as the resulting policy can still run using only local information at execution time. It's worth being honest that agent modeling (9.6) and self-play (9.8) fit this framing a little more loosely than the others — flagging that nuance rather than forcing a clean narrative shows the room you've thought carefully about the material rather than just pattern-matching a summary slide.

## Slide 21 — From Chapter 6 to Chapter 9: A Recurring Pattern

**Slide content:**
- Chapter 6 (tabular): fictitious play, JAL, minimax-Q, Bayesian agent modeling
- Chapter 9 (deep): the *same ideas*, scaled up with neural network function approximation
- Pattern: Chapter 9 is not a new theory of multi-agent learning — it's Chapter 6's theory, made practical for large/continuous state spaces

**Speaker notes:**
This is a nice unifying observation for the semester as a whole, not just today's sections: fictitious play and JAL-AM from Chapter 6.3 become deep JAL-AM in 9.6; minimax-Q's underlying zero-sum-game reasoning becomes the motivation for self-play and MCTS in 9.8; and so on. If your seminar covered Chapter 6 earlier in the semester, this is a great moment to explicitly ask the room to recall specific Chapter 6 algorithms and connect them live — it reinforces retention of the earlier material and makes Chapter 9 feel less like a grab-bag of separate deep learning tricks.

## Slide 22 — What the Book Deliberately Left Out

**[Added context]**

**Slide content:**
- Communication learning between agents (explicitly noted as out of scope in the book's preface)
- Evolutionary game theory approaches to multi-agent learning
- Worth knowing these exist, even if outside this book's coverage

**Speaker notes:**
Nice, honest closing note — shows intellectual completeness without needing you to actually teach these topics. The book's own preface explicitly flags that it doesn't cover learned communication protocols between agents (how agents might learn to send each other messages, robustly, over noisy channels) or evolutionary-game-theoretic approaches to multi-agent learning, pointing interested readers to other surveys instead. Mentioning this signals to the room that Chapter 9, comprehensive as it feels, is a curated subset of a much larger active research area — a good note to end a semester-long book study on.

## Slide 23 — Closing / Discussion

**Slide content:**
- Thank you slide
- Open floor: any final questions on Chapter 9, or on the semester's material as a whole

**Speaker notes:**
Since this is the last presentation of the semester, it's worth explicitly inviting broader questions here, not just about today's sections — people may want to ask something that ties back to a much earlier week. Thank the room, thank Seong Jin, and open the floor.

---

## Optional bonus slide (use only if you want it — not required content)

If you want a lighter closing anecdote tying the whole semester together, you could briefly mention how even with everything covered in Chapter 9 (value decomposition, agent modeling, parameter sharing, self-play, population-based training), real-world multi-agent problems like full 11-a-side football or RoboCup's 2050 goal remain far beyond current methods — a nice one-line reminder that the chapter's tools are powerful but the field's hardest open problems are still open. This is entirely optional and tangential to the syllabus content, so only include it if it fits your and Seong Jin's taste for how to end the session — it wasn't part of your assigned sections and the room may not have the shared context for it unless you set it up quickly.

---

## Anticipated audience questions (prep these answers in advance)

1. **"If parameter sharing is so cheap and effective, why would you ever bother with value decomposition (VDN/QMIX) instead?"**
   They solve different problems. Parameter sharing helps with sample efficiency when agents are similar; it does nothing by itself to solve credit assignment in genuinely difficult coordination problems. You can (and often do) combine both — e.g., share parameters across agents' individual utility networks in QMIX.

2. **"Doesn't experience sharing risk teaching an agent to imitate bad habits from a struggling teammate?"**
   This is exactly what the importance-sampling-style correction is meant to guard against — poor-fit experience (unlike what your own current policy would do) gets down-weighted rather than blindly copied. It's not a perfect fix, and this is a fair place to admit the technique has limits, especially early in training when all agents' policies are still bad and hard to distinguish from each other.

3. **"Which technique from 9.7 would you use for something like RoboCup/football-style agents?"**
   Good bridge question if it comes up, since it lets you reference earlier semester material: football-playing agents on the same team are a good example of *weakly* homogeneous agents — same action/observation interface, but real specialization by position (defenders vs. attackers) is essential. That points toward experience sharing (or selective/clustered parameter sharing) rather than full parameter sharing, since forcing one identical network onto every position would likely wash out exactly the specialization that makes the strategy work.
