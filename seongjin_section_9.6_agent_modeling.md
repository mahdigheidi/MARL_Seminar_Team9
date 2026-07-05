# Section 9.6 — Agent Modeling with Neural Networks
### Presentation Guide for Seong Jin
**Slot length:** ~40 minutes (hard ceiling 45 min) | **Position in session:** second speaker, right after the opening introduction

---

## How to use this document

This document gives you a slide-by-slide outline plus full speaker notes for Section 9.6. It's built directly from the book's own structure (9.6.1 Joint-Action Learning with Deep Agent Models, 9.6.2 Learning Representations of Agent Policies), plus some extra context pulled from the wider literature to make the story land better for the room. Anything marked **[Book figure]** refers to an actual figure in the book — pull up the real figure/numbers from your copy when you present, since I'm describing what the figure shows conceptually rather than quoting exact curves or values. Anything marked **[Added context]** is supplementary material from outside the book — useful for depth and for answering audience questions, but optional if you're tight on time.

**Suggested timing:** ~18 slides, ~2.2 min/slide average. Slides 1, 3, and 18 are quick (title/motivation/summary); slides 8–9 and 14 are the technical heart and deserve the most airtime.

---

## Learning objectives for the audience

By the end of your section, the room should be able to:
1. Explain *why* explicitly modeling other agents is a distinct strategy from both independent learning (9.3) and centralized critics/value decomposition (9.4–9.5).
2. Describe how Joint-Action Learning with Agent Modeling (JAL-AM) scales from the tabular setting (Ch. 6.3) to deep neural networks.
3. Explain the difference between predicting an agent's *next action* versus learning a *representation* of an agent's policy — and why the latter is more general.
4. Connect agent modeling back to the "non-stationarity" problem from Chapter 5, and forward to Seong Jin's teammate's topic (parameter sharing assumes homogeneity — agent modeling is partly what you'd need when agents *aren't* homogeneous).

---

## Slide 1 — Title Slide

**Slide content:**
- "Agent Modeling with Neural Networks" — Section 9.6
- Your name, course, date

**Speaker notes:**
Brief self-intro if not already covered by your teammate's opening (10 seconds), then straight into the roadmap: "In this section we're going to look at what happens when, instead of treating other agents as background noise in the environment, we build an agent that actively tries to predict what its teammates or opponents are going to do next."

---

## Slide 2 — Recap: Where We Are in the Book

**Slide content:**
- Chapter 9 so far: 9.3 Independent Learning → 9.4 Centralized Critics → 9.5 Value Decomposition
- Today: 9.6 Agent Modeling with Neural Networks
- Callback: Chapter 6.3 already covered "Agent Modeling" — fictitious play, JAL with agent modeling, Bayesian learning (tabular)

**Speaker notes:**
Remind the room that "agent modeling" isn't a new idea introduced in Chapter 9 — it's a *return* to an idea from Chapter 6.3, which covered fictitious play and tabular joint-action learning with agent modeling (JAL-AM). The novelty in 9.6 is scaling those same ideas up with neural networks so they work in environments with large or continuous observation spaces, where you can't just keep a lookup table of "how often has this agent played each action in each state." This framing helps the room see Chapter 9 as "Chapter 6's ideas, but with function approximation" rather than a totally separate set of algorithms — which is true for almost every section in Chapter 9, not just this one.

---

## Slide 3 — Motivation: Why Model Other Agents At All?

**Slide content:**
- Independent learning (9.3) treats other agents as part of the environment → non-stationarity
- Centralized critics (9.4) and value decomposition (9.5) fix credit assignment during *training*, but at execution time each agent still acts on local info only
- Agent modeling asks a different question: **can an agent explicitly predict what others will do, and use that prediction to act better?**

**Speaker notes:**
This is the key distinguishing idea for your whole section, so spend real time here. Independent learners (Section 9.3) are technically "solving" a moving-target problem without acknowledging it — from any one agent's point of view, the environment's dynamics appear to be changing over time purely because the other agents' policies are changing too. Centralized critics and value decomposition (which your classmates covered last time / earlier in this chapter) address the *credit-assignment* half of the multi-agent problem, but they don't require any agent to actually understand or predict what its teammates are individually likely to do. Agent modeling is orthogonal to both of these — it's specifically about building an explicit, learned model of another agent's behavior and using it as an input to your own decision-making. You can combine it with either independent learning or CTDE-style training.

---

## Slide 4 — What Counts as "Agent Modeling"? (Broader Picture)

**Slide content:**
- Policy reconstruction — predict an agent's action probabilities directly from observed behavior
- Type-based reasoning — maintain a belief over a small set of candidate "types" of agent
- Recursive/"theory of mind" reasoning — model that the other agent is *also* modeling you
- Section 9.6 focuses mainly on **policy reconstruction**, scaled up with neural networks

**[Added context]**
**Speaker notes:**
This slide is extra context beyond the book, but it's worth including because it stops the room from thinking "agent modeling" is one single technique. This taxonomy comes from Albrecht & Stone's 2018 survey *"Autonomous Agents Modelling Other Agents"* (Stefano Albrecht — one of your book's own authors) — it's essentially the canonical reference for this whole subfield, and it identifies several distinct families of methods for one agent to reason about others. Policy reconstruction is the most direct: literally trying to predict what action another agent will take next given what you've observed of it so far. Type-based reasoning is more like the Bayesian agent-modeling in Chapter 6.3.3, where instead of predicting continuous action probabilities you maintain a belief distribution over a small number of discrete hypotheses about what "type" of agent you're dealing with. Recursive reasoning (theory of mind, "I think that you think that I think...") is the deepest and least scalable of these — it shows up in things like the Recursive Modeling Method and later work like PR2/GR2. Section 9.6 of your book sits squarely in the "policy reconstruction" family, just implemented with neural networks instead of tables or hand-designed models.

---

## Slide 5 — From Tabular to Deep: What Actually Needs to Scale?

**Slide content:**
- Tabular JAL-AM (Ch. 6.3.2): keep counts/frequencies of other agents' actions per state → works fine for small, discrete state spaces
- Problem: real environments have huge or continuous observation spaces — no way to keep a table of "what does agent *j* do in this state"
- Solution: replace the table with a neural network that maps *observations/histories → predicted action distribution*

**Speaker notes:**
Bridge slide. In the tabular setting from Chapter 6, an agent could literally keep a frequency count of how often another agent chose each action in each state — that's essentially what fictitious play and tabular JAL-AM do. The moment your state space is an image, a large grid, or a continuous vector, that approach breaks down completely — there just aren't enough visits to each exact state to build a reliable frequency table. The natural fix, exactly the same move the book makes going from Chapter 6 to Chapter 8/9 for value functions, is to swap the table for a function approximator: a neural network trained to output a predicted action distribution for another agent given that agent's observation history.

---

## Slide 6 — 9.6.1: Joint-Action Learning with Deep Agent Models (Deep JAL-AM)

**Slide content:**
- Each agent *i* maintains an **agent model network** for each other agent *j*: π̂ⱼ(a | oⱼ) — approximates agent j's policy
- Trained via supervised learning: maximize the likelihood of *j*'s actually observed actions
- Agent *i*'s own action-value estimation is computed using the *predicted* joint action rather than agent i's action alone

**Speaker notes:**
This is the deep-learning version of Section 6.3.2's tabular Joint-Action Learning with Agent Modeling. The setup: agent *i* doesn't just learn its own policy; it also learns a small model network for each other agent it interacts with. That model network takes agent *j*'s observation (or observation history, if partial observability matters) as input, and outputs a predicted probability distribution over agent *j*'s actions. Critically, this model is trained the boring, reliable way — plain supervised learning, maximizing the likelihood of whatever action agent *j* was actually observed to take. There's no RL involved in training the agent model itself. Once you have this predicted distribution for every other agent, agent *i* can compute its own action-values as an *expectation* over the likely actions of everyone else, rather than being surprised by them — this is the "joint-action learning" part, since you're reasoning over the full joint action space instead of just your own marginal action.

---

## Slide 7 — Deep JAL-AM: How the Prediction Gets Used

**Slide content:**
- Agent *i*'s Q-function is estimated as: Q(s, aᵢ) ≈ Σ over predicted joint actions of other agents, weighted by their predicted probabilities
- Effectively: agent *i* computes an **expected best response** to its (learned) belief about what everyone else will do
- Contrast with independent learning: independent learners have no explicit belief at all — they just observe outcomes and hope for the best

**Speaker notes:**
Make this concrete with the mechanics: once agent *i* has a predicted distribution over agent *j*'s next action (and similarly for every other agent), it can plug those predicted probabilities into its own value estimation instead of only using the one action that agent *j* actually happened to take at training time. This is exactly analogous to the tabular fictitious-play idea from Chapter 6: fictitious play has each agent best-respond to the empirical *frequency* of opponents' past actions; deep JAL-AM does the same thing but the "frequency table" has been replaced by a trained neural network that can generalize across similar-looking observations rather than needing to see the exact same state repeated many times.

---

## Slide 8 — Example: Level-Based Foraging (Deep JAL-AM vs. IDQN)

**[Book figure — Figure 9.20]**

**Slide content:**
- Environment: level-based foraging (recurring example throughout the book)
- Compares: Independent DQN (IDQN, no agent modeling) vs. Deep JAL-AM
- Bring up your copy of Figure 9.20 here and walk through the actual learning curves

**Speaker notes:**
This is a good slide to actually project the real figure from the book rather than recreate it from memory, since the specific numbers/curve shapes matter here. The high-level story to set up for the room before you show the curve: level-based foraging requires implicit coordination — an item can only be collected if the *right combination* of robots shows up next to it with enough combined skill level. Because deep JAL-AM lets each agent factor in a prediction of what its teammates are likely to do, you'd expect it to learn faster and/or reach a better final policy than IDQN, which has no way to anticipate coordination and instead just experiences its teammates' behavior as unpredictable noise. Use the real figure to confirm (or complicate) that story for the room — if the actual curves show something more nuanced, that's an even better discussion point.

---

## Slide 9 — Limitations of Deep JAL-AM

**Slide content:**
- Assumes each agent can **observe** other agents' actions — not always true (partial observability, communication constraints)
- Scaling: needs one agent-model network *per other agent* → grows with team size
- Still doesn't solve non-stationarity completely: the other agents are *also* learning and changing, so your model of them constantly needs retraining
- Purely reactive: models what others currently do, not what they'll do once they *also* adapt to you (no recursive reasoning)

**Speaker notes:**
Don't let this section sound like a silver bullet — this is a good place to invite some skepticism from the room, and it sets up a nice contrast with your teammate's/your own upcoming sections. First, deep JAL-AM needs to actually observe other agents' actions, which isn't guaranteed in every environment (compare to Google Research Football, discussed in Ch. 11, where you don't get to see the opposing team's "intended" action, only the resulting state). Second, it scales linearly (at best) in the number of other agents you're modeling — each teammate needs its own model network, so this doesn't obviously scale to something like an 11-a-side team. Third — and this is the subtle one — deep JAL-AM only partially solves non-stationarity: your model of agent *j* has to keep being retrained as agent *j*'s own policy keeps changing during training, so you've converted "non-stationary environment" into "non-stationary supervised-learning target," which is progress but not a full fix.

---

## Slide 10 — Transition: From Predicting Actions to Predicting *Policies*

**Slide content:**
- Deep JAL-AM predicts: "what action will agent *j* take right now?"
- 9.6.2 asks a more general question: "what *kind* of policy does agent *j* have, in general?"
- Motivation: a reusable representation is more sample-efficient than re-predicting from scratch every single timestep

**Speaker notes:**
Use this as your pivot slide into 9.6.2. The limitation you just described — needing to constantly re-predict raw action probabilities — motivates a different idea: instead of predicting the next action directly, learn a compact, general-purpose *representation* of what kind of policy an agent has. Think of the difference between "guessing this specific chess move" versus "recognizing this is an aggressive, sacrifice-heavy player" — the second is a much more compact and reusable piece of information that still lets you make good predictions in situations you haven't seen before.

---

## Slide 11 — 9.6.2: Learning Representations of Agent Policies

**Slide content:**
- Core idea: train an **encoder** that maps an agent's observed trajectory → a low-dimensional embedding vector
- Train a **decoder** that reconstructs/predicts that agent's actions from the embedding
- The embedding becomes a compact, general summary of "what kind of agent this is"

**[Book figure — Figure 9.21: Encoder–decoder architecture]**

**Speaker notes:**
This is the technical core of your presentation — take your time here. The architecture is a fairly classic encoder–decoder setup, similar in spirit to autoencoders you may have seen elsewhere in ML, but applied to *behavior* rather than images or text. The encoder watches a stretch of an agent's observed trajectory (its sequence of observations and/or actions) and compresses it down into a small embedding vector. The decoder's job during training is to take that embedding and predict the agent's actions — this prediction task is what forces the embedding to actually capture meaningful information about the agent's behavior, rather than collapsing to something trivial. Once trained, you can discard the decoder and just use the encoder as a standalone "agent typing" module: feed it a new agent's recent trajectory, get back an embedding that summarizes its behavioral style.

---

## Slide 12 — Using the Learned Representation

**Slide content:**
- The embedding gets concatenated into your **own** policy/value network's input, alongside your own observation
- Lets your policy condition its behavior on "what kind of teammate/opponent am I dealing with," without re-deriving that from scratch every step
- Reusable across episodes and even across previously unseen agents with similar behavioral styles

**Speaker notes:**
Emphasize the generalization benefit here, since it's the main payoff versus deep JAL-AM. Because the embedding is a general-purpose summary rather than a raw next-action prediction, it can generalize to *new* agents whose behavior resembles agents seen during training, even without observing that exact new agent for very long. This is conceptually close to the "type-based reasoning" idea from Chapter 6.3.3 (Bayesian belief over discrete types), except here the "types" aren't hand-specified in advance — they're learned automatically as points in a continuous embedding space.

---

## Slide 13 — Example: Representation-Based Agent Models with a Centralized Critic

**[Book figure — Figure 9.22]**

**Slide content:**
- Environment: level-based foraging (again — good continuity for the audience)
- Compares centralized A2C **with** vs **without** representation-based agent models
- Pull up the real figure and walk through what it shows

**Speaker notes:**
Same approach as Slide 8 — bring up the actual figure rather than reconstructing exact numbers from memory. Frame the expected story before revealing the curve: if the learned representations genuinely capture useful information about teammates, adding them as extra input to a centralized critic should improve sample efficiency or final performance relative to the same critic without that extra information. Since this experiment layers agent modeling *on top of* a centralized critic (9.4's contribution), this is also a nice moment to point out that these Chapter 9 techniques aren't mutually exclusive — you can combine centralized critics, value decomposition, agent modeling, and parameter sharing in the same system, and real implementations often do.

---

## Slide 14 — Bigger Picture: Why This Matters Beyond the Book

**[Added context]**

**Slide content:**
- Connects to "Theory of Mind" research in cognitive science (modeling others' beliefs/intentions)
- Connects to opponent modeling in competitive games (predicting what a specific opponent, not just "any" opponent, will do)
- Connects forward to **parameter sharing** (next section!) — parameter sharing *assumes* agents are similar; agent modeling is one way to detect/handle it when they're *not*

**Speaker notes:**
This slide is your bridge to the next presenter, so make it land. Learned agent embeddings show up again in a completely different context later in the MARL literature: the "Selective Parameter Sharing" line of work (Christianos, Papoudakis, Rahaman & Albrecht — note this involves two of your own book's authors) uses almost exactly this kind of learned embedding to automatically *cluster* agents into groups before deciding which agents should share network parameters. In other words: the representation-learning idea from 9.6.2 isn't just useful for modeling opponents — it's also one natural way to automatically decide whether a given set of agents is homogeneous enough to benefit from the parameter-sharing techniques that your teammate's about to cover in Section 9.7. That's a good explicit segue: "So now that we've seen how to model when agents behave differently, let's look at what we can do when agents behave similarly — over to [name] for Section 9.7."

---

## Slide 15 — Recap Table: Two Flavors of Agent Modeling in 9.6

**Slide content:**

| | 9.6.1 Deep JAL-AM | 9.6.2 Representation Learning |
|---|---|---|
| Predicts | Next action of agent *j* | General embedding of agent *j*'s policy |
| Training signal | Supervised (max-likelihood on observed actions) | Encoder–decoder reconstruction/prediction loss |
| Used for | Expected best-response Q-values | Extra conditioning input to own policy/critic |
| Generalizes to new agents? | Poorly — retrained per agent | Better — embedding space can generalize |

**Speaker notes:**
Use this table to consolidate before your summary slide — it's an easy one to talk through quickly and gives the room something concrete to remember. Emphasize the "generalizes to new agents" row in particular, since that's the throughline of why the book presents these two ideas in this order — it's a progression from a more direct but narrower technique (deep JAL-AM) toward a more abstract but more broadly useful one (representation learning).

---

## Slide 16 — Discussion Prompt (Optional, if time allows)

**Slide content:**
- "If deep JAL-AM requires observing others' actions, how might you approach agent modeling in a game where you can *only* see the resulting state, not the action that caused it (e.g., Google Research Football)?"

**Speaker notes:**
Good filler/engagement slide if you're running slightly ahead of schedule, or a good way to end with an open question rather than just stopping. There's no single "book answer" here — it's meant to get people talking. A reasonable direction to nudge the discussion toward if it stalls: you could try to infer the action from the state transition (inverse-dynamics-style modeling), or you could sidestep the problem entirely and predict the *resulting state* rather than the specific action, which is a strictly weaker but more observable target.

---

## Slide 17 — Section 9.6 Summary

**Slide content:**
- Agent modeling = building an explicit, learned model of other agents rather than treating them as environment noise
- Two techniques: predicting **actions** (deep JAL-AM) vs. predicting a general **representation** of policy
- Addresses non-stationarity and coordination — orthogonal to, and combinable with, centralized critics/value decomposition
- Sets up the assumption that Section 9.7 will build on: what if agents are similar enough that you don't need to model each one individually?

**Speaker notes:**
Quick, clean recap — don't introduce new information here. This is also your literal handoff moment, so make sure the transition sentence is prepared: something like, "That's Section 9.6 — now [user's name] is going to pick up right where that last point left off, looking at what we can do when agents *are* similar enough to share information directly, in Section 9.7."

---

## Slide 18 — Handoff Slide

**Slide content:**
- "Thank you — over to [user's name] for Section 9.7: Environments with Homogeneous Agents"

**Speaker notes:**
Keep this short — just a clean, confident handoff. No need to summarize again.

---

## Anticipated audience questions (prep these answers in advance)

1. **"How is agent modeling different from just using a centralized critic?"**
   A centralized critic (9.4) gives the *critic* privileged access to joint information *during training only*, to fix credit assignment — the actor still doesn't know or predict what others will do. Agent modeling gives an agent an explicit, learned prediction about others' behavior that can be used at execution time too (if you allow it access to enough observation), and it's really about coordination/non-stationarity, not credit assignment per se.

2. **"Doesn't this just move the non-stationarity problem — now your model of the other agent has to keep adapting?"**
   Yes, and it's worth saying so plainly if asked — this is a genuinely open limitation. It converts "non-stationary environment" into "non-stationary supervised learning target for the agent model," which is often easier to manage (since you at least know the training signal is a plain classification/regression problem) but doesn't eliminate the underlying issue.

3. **"Could you use agent modeling in a zero-sum competitive setting, like the self-play section (9.8)?"**
   Yes in principle — recursive/theory-of-mind style agent modeling is closely related to exploitability analysis in game theory. This isn't the book's main framing (9.8 relies on search plus self-play rather than explicit opponent modeling), but it's a fair connection to draw if the discussion goes there.

---

## Suggested one-line callback for the very end of the whole session

If there's a moment during the final Chapter 9 summary to reference back to your section, a good one-liner is: *"Section 9.6 was about handling agents that are different from you; Section 9.7 is about exploiting agents that are the same as you — together they cover both ends of the homogeneity spectrum that Chapter 9 cares about."*
