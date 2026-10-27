# Next best moves do not converge (open issue)

Status: reported by the creator on 2026-10-21; not fixed. Fix later.

## Symptom

Applying the "Next best moves" in step 4 one after another does not lead to a
stable plan. There is always another move, and where you end up depends on
where you started.

## What a scripted walk shows (sample scenario, cloud on)

The walk applies the single best move each step (the top swap or the top "add
GPU", whichever predicts more margin):

- **Predicted gains are accurate one step at a time.** For example, A's first
  swap predicted +$6,052/day and delivered +$6,053.
- **Margin rises at every step; no A→B→A cycle was seen.**
- **Config A never stops.** After 10 moves it is still adding GPUs and swapping
  models: Qwen 3.5 → Muse Glimmer 3 → Laguna S 2.1. Margin goes from $18.0k
  to $30.9k a day. GPU cost is small next to the value served, so "add 1 GPU"
  stays positive almost indefinitely. Nothing caps the pool or asks whether
  the gain is still worth it.
- **Config B stops after 6 moves, at $24.2k a day.** The same catalog and
  workload, started from different models, settles somewhere else: greedy
  search finds a local optimum.
- **Gains are not monotone.** For Muse Glimmer 3, +1 GPU predicted +$351, then
  the next +1 GPU predicted +$948. GPU counts that suit a layout (TP/DP
  multiples) create steps, so one-GPU-at-a-time search can stall in front of
  a better point.
- **Swaps create duplicate deployments.** Two separate Qwen 3.5 deployments
  end up side by side instead of one larger one. Merging them is not a move
  the search can make.
- **Swaps and "add GPU" are scored separately,** each against the current
  plan. Their gains don't add up, and the list never shows a combined move.

## Seen in the UI after applying A's top move

- The three swap rows all lose money (−$1,919, −$1,933, −$2,437 a day). They
  are still listed first, and the first one keeps the highlighted Apply
  button. Swaps and GPU additions are listed in fixed slots (3 + 2), not
  ranked together, and losing moves are not filtered out.
- "+1 GPU → Qwen 3.5 35B" appears twice, once for each of the two duplicate
  Qwen 3.5 deployments.

## Directions to consider (none chosen)

1. Stop rule: show a move only when its gain clears a threshold, for example
   a minimum $/day or a payback period per added GPU. Say "plan is stable"
   when none does.
2. Budget: a GPU budget or maximum pool size, so "add GPU" stops at a real
   constraint.
3. Search: score multi-step plans (beam search, or a swap + grow pair) rather
   than single greedy steps. Allow merging identical deployments.
4. Report the destination: show where repeated application would end
   ("following all moves: +$12.9k/day with 9 GPUs"), not only the next step.

## How to reproduce

Use `planner_service.apply_model_swap` / `apply_gpu_expansion` together with
`presentation.econ.compute_swap_recs` and the projection's `recommendations`.
Loop "apply the best positive move" on `create_default_scenario()` A and B,
and print margin and deployments at each step.
