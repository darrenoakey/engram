# =============================================================================
#  dream — the nightly consolidation, the only durable individuation writer (§5)
#  why: the guards that keep online learning safe (recall + sentinels) are
#  distributional — on a batch of one turn they are noise, so a durable write is
#  only defensible over a night's batch. dream corroborates each unconsolidated
#  experience, self-edits survivors into assistant-knowledge, absorbs them into
#  the OVERLAY (base consolidation stays a separate existing op), then health-
#  gates the whole night: commit atomically or restore the overlay bit-for-bit.
#  The night is one revertible unit — including the probes it optimistically added.
# =============================================================================
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone

from individuation import atomize, corroborate, selfedit
from individuation.probe import FactProbe, sentinels

_STOPWORDS = frozenset({
    "the", "user", "you", "your", "and", "for", "with", "that", "this", "are", "was", "were",
    "has", "have", "had", "his", "her", "its", "our", "their", "who", "not", "but", "they",
})


# ##################################################################
# dream report
# the night's outcome: whether it committed, how many facts were learned versus
# dropped, and the health numbers the gate ruled on
@dataclass
class DreamReport:
    committed: bool
    facts_learned: int
    dropped: int
    recall: float
    entropy: float
    sycophancy: float


# ##################################################################
# dream
# consolidate the day's unconsolidated experiences into the overlay under a
# health gate; commit atomically when recall and sentinels pass, else revert
def dream(host, overlay, updater, journal, experience_log, individuation_probe, config) -> DreamReport:
    exps = experience_log.unconsolidated()
    if not exps:
        return DreamReport(False, 0, 0, 0.0, 0.0, 0.0)
    probe_floor = len(individuation_probe.all())
    try:
        with host.gpu_lock:
            snapshot = overlay.snapshot()
        facts, dropped = _absorb_all(host, overlay, updater, journal, individuation_probe, config, exps)
        report = individuation_probe.run(host)
        sent = sentinels(host, config)
        recall_ok = report.recall >= config.individuation.probe_recall_target
        return _settle(host, overlay, journal, experience_log, individuation_probe, exps, snapshot,
                       probe_floor, facts, dropped, report, sent, recall_ok)
    finally:
        host.model.eval()


# ##################################################################
# absorb all
# corroborate then self-edit then absorb each experience; durable survivors train
# the overlay and grow the probe set, everything else is journaled and dropped
# only genuine facts and preferences earn a durable write. Gating on KIND as well
# as the durable flag is deliberate: a weak classifier will sometimes mistag a
# role-play or hypothetical as durable, but it labels the KIND reliably, so a
# role_play/hypothetical/transient/command never persists even when durable is wrong
_LEARNABLE_KINDS = frozenset({"fact", "preference"})


# ##################################################################
# repolish
# the low-speed refresher for ALREADY-learned facts that have gone stale: re-
# synthesize each stale probe's QA from its question (the probe IS the fact —
# self-edit is regenerable from a question), absorb under the consolidate kind at
# gentler repolish_epochs, then health-gate identically to a dream. On commit the
# re-trained probes are timestamped; on revert the overlay is restored and the
# probes are left untouched (the fact was already learned — worst case it is not
# refreshed this pass). No experience-log change: these facts were consolidated
# long ago. This is what makes learning CONTINUOUS rather than one-shot
def repolish(host, overlay, updater, journal, individuation_probe, config, probes) -> DreamReport:
    if not probes:
        return DreamReport(False, 0, 0, 0.0, 0.0, 0.0)
    try:
        with host.gpu_lock:
            snapshot = overlay.snapshot()
        trained = _repolish_all(host, overlay, updater, journal, config, probes)
        report = individuation_probe.run(host)
        sent = sentinels(host, config)
        recall_ok = report.recall >= config.individuation.probe_recall_target
        return _settle_repolish(host, overlay, journal, individuation_probe, trained, snapshot,
                                report, sent, recall_ok)
    finally:
        host.model.eval()


# ##################################################################
# repolish all
# re-synthesize QA from each stale probe's question and absorb the answer tokens
# under the consolidate kind. The probe's expect-word is the fact; the question is
# a paraphrase that does not leak it, so self-edit produces fresh assistant-
# knowledge answers. A probe that fails to synthesize is skipped (dropped), not
# fatal — the fact stays learned, it just is not refreshed this pass
def _repolish_all(host, overlay, updater, journal, config, probes) -> list[FactProbe]:
    trained: list[FactProbe] = []
    for probe in probes:
        pairs = selfedit.synthesize(host, _statement_from_probe(probe), config)
        if not pairs:
            continue
        _train_fact(host, overlay, updater, journal, pairs, config.individuation.repolish_epochs)
        trained.append(probe)
    return trained


# ##################################################################
# statement from probe
# the input self-edit expects a statement of fact; from a stored probe we have a
# question + its expected answer word. Re-state it as "The user <answer-related>
# <question subject>" is overkill — self-edit only needs a seed it can paraphrase
# into QA, and the probe's own question plus its expect word reconstructs the fact
# plainly enough for the reformulator
def _statement_from_probe(probe: FactProbe) -> str:
    return f"{probe.question} {probe.expect}."


# ##################################################################
# settle repolish
# the re-polish health gate: on pass, timestamp the re-trained probes and journal
# a repolish; on fail, restore the overlay and journal a repolish_reverted. Probes
# are never added or removed here (they pre-existed), so there is no truncate
def _settle_repolish(host, overlay, journal, individuation_probe, trained, snapshot, report, sent,
                     recall_ok) -> DreamReport:
    healthy = recall_ok and sent.healthy
    dropped = len(trained) if not healthy else 0
    outcome = DreamReport(healthy, len(trained), dropped, report.recall, sent.entropy, sent.sycophancy)
    fields = {"facts": len(trained), "dropped": dropped, "recall": report.recall,
              "entropy": sent.entropy, "sycophancy": sent.sycophancy}
    if healthy:
        when = _now_iso()
        individuation_probe.touch([p.question for p in trained], when)
        journal.record("repolish", **fields)
        return outcome
    with host.gpu_lock:
        overlay.restore(snapshot)
    journal.record("repolish_reverted", reason=_reason(recall_ok, sent.healthy), **fields)
    return outcome


def _absorb_all(host, overlay, updater, journal, individuation_probe, config, exps) -> tuple[int, int]:
    facts = 0
    dropped = 0
    for exp in exps:
        # split a multi-fact turn into atomic statements first, then corroborate
        # each atom independently — a compound turn ("my wife is Arlene, my son is
        # Leo") otherwise collapses to one generic statement and loses every proper
        # noun. A single-fact turn yields one atom ([user_text]) and is unchanged
        atoms = atomize.atomize(host, exp.user_text, config)
        if len(atoms) > 1:
            journal.record("atomize", experience_id=exp.id, count=len(atoms), atoms=atoms)
        for statement_seed in atoms:
            verdict = corroborate.classify(host, statement_seed, config)
            learnable = verdict.durable and verdict.kind in _LEARNABLE_KINDS
            pairs = selfedit.synthesize(host, verdict.statement, config) if learnable else []
            if not learnable or not pairs:
                dropped += 1
                journal.record("experience", experience_id=exp.id, durable=verdict.durable, kind=verdict.kind)
                continue
            _train_fact(host, overlay, updater, journal, pairs, config.individuation.dream_epochs)
            individuation_probe.add(_fact_probe(pairs))
            facts += 1
    return facts, dropped


# ##################################################################
# fact probe
# build the cold-recall check for a learned fact. The expected word is drawn from
# the ANSWER the model was trained to produce (not the raw user statement, whose
# longest word is often the wrong target for a compound sentence), and the
# question is a paraphrase that does NOT itself contain that word, so recall is
# genuine rather than a question that leaks its own answer
def _fact_probe(pairs) -> FactProbe:
    for qa in pairs:
        noun = _key_noun(qa.answer)
        if noun and noun not in qa.question.lower():
            return FactProbe(question=qa.question, expect=noun)
    qa = pairs[-1]
    return FactProbe(question=qa.question, expect=_key_noun(qa.answer))


# ##################################################################
# train fact
# absorb every QA pair of one fact into the overlay: an absorb update teacher-
# forces the assistant-knowledge ANSWER tokens under the guarded pipeline
def _train_fact(host, overlay, updater, journal, pairs, epochs: int) -> None:
    examples = [selfedit.training_example(host, qa) for qa in pairs]
    for _ in range(max(1, epochs)):
        for token_ids, gen_start, span in examples:
            with host.gpu_lock:
                updater.apply(host.model, overlay, token_ids, gen_start, [span], 1.0, "consolidate", None, journal)


# ##################################################################
# settle
# the atomic health gate: on pass, mark the whole night consolidated and journal
# a dream; on fail, restore the overlay and the probe set and journal a revert
def _settle(host, overlay, journal, experience_log, individuation_probe, exps, snapshot, probe_floor,
            facts, dropped, report, sent, recall_ok) -> DreamReport:
    healthy = recall_ok and sent.healthy
    outcome = DreamReport(healthy, facts, dropped, report.recall, sent.entropy, sent.sycophancy)
    fields = {"facts": facts, "dropped": dropped, "recall": report.recall,
              "entropy": sent.entropy, "sycophancy": sent.sycophancy}
    if healthy:
        experience_log.mark_consolidated([exp.id for exp in exps])
        journal.record("dream", **fields)
        return outcome
    with host.gpu_lock:
        overlay.restore(snapshot)
    individuation_probe.truncate(probe_floor)
    journal.record("dream_reverted", reason=_reason(recall_ok, sent.healthy), **fields)
    return outcome


# ##################################################################
# reason
# a short human reason a night was reverted — low cold recall, unhealthy
# sentinels, or both — recorded on the dream_reverted event
def _reason(recall_ok: bool, sent_healthy: bool) -> str:
    if not recall_ok and not sent_healthy:
        return "recall_and_sentinels"
    return "recall_below_target" if not recall_ok else "sentinels"


# ##################################################################
# now iso
# a UTC ISO timestamp; used to stamp a re-polished fact's last_trained_at
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ##################################################################
# key noun
# the most salient content word of a statement, used as the probe's expected
# substring (e.g. "The user is allergic to shellfish." -> "shellfish")
def _key_noun(statement: str) -> str:
    words = re.findall(r"[A-Za-z]+", statement.lower())
    candidates = [w for w in words if w not in _STOPWORDS and len(w) > 2]
    if not candidates:
        return statement.strip()[:20]
    return max(candidates, key=len)
