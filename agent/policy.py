"""
agent/policy.py — Hybrid Policy Agent (Rule-based + optional LLM via Claude API)
Set ANTHROPIC_API_KEY env var to enable LLM reasoning.
"""
import os, json
import numpy as np
from dataclasses import dataclass, field

@dataclass
class EmbeddingPolicy:
    sensitivity_threshold: float
    max_bpp:               float
    allowed_data_types:    list
    skip_roi:              bool
    compliance:            list
    rationale:             str
    region_plan:           dict = field(default_factory=dict)
    llm_used:              bool = False

MODALITY_CFG = {
    "xray": {"threshold": 0.45, "max_bpp": 1.0},
    "mri":  {"threshold": 0.35, "max_bpp": 0.8},
    "ct":   {"threshold": 0.40, "max_bpp": 0.9},
}
CONSENT_CFG = {
    "full":       ["anonymized_id","study_notes","diagnosis_tags","timestamp"],
    "anonymized": ["anonymized_id","timestamp"],
    "none":       [],
}

def _rule_policy(modality, consent, sensitivity_map):
    cfg     = MODALITY_CFG.get(modality.lower(), MODALITY_CFG["xray"])
    allowed = CONSENT_CFG.get(consent, [])
    return EmbeddingPolicy(
        sensitivity_threshold=cfg["threshold"], max_bpp=cfg["max_bpp"],
        allowed_data_types=allowed, skip_roi=True,
        compliance=["HIPAA","GDPR"],
        rationale=(f"Modality '{modality}': threshold={cfg['threshold']}. "
                   f"Consent '{consent}' permits {allowed}. ROI pixels skipped."),
        region_plan={"critical_roi":{"embed_rate":0.0},"background":{"embed_rate":cfg["max_bpp"]}},
        llm_used=False,
    )

def _llm_policy(modality, consent, sensitivity_map):
    api_key = os.environ.get("ANTHROPIC_API_KEY","")
    if not api_key:
        print("[agent] No API key — rule-based policy")
        return _rule_policy(modality, consent, sensitivity_map)
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        stats  = {"mean":float(sensitivity_map.mean()), "pct_safe":float((sensitivity_map<0.4).mean()*100)}
        prompt = f"""Medical RDH policy expert. Modality={modality}, Consent={consent}, Sensitivity={json.dumps(stats)}.
Respond ONLY with JSON: {{"sensitivity_threshold":<float>,"max_bpp":<float>,"allowed_data_types":[...],"rationale":"<2 sentences>"}}"""
        resp = client.messages.create(model="claude-sonnet-4-20250514", max_tokens=300,
                                      messages=[{"role":"user","content":prompt}])
        data = json.loads(resp.content[0].text.strip())
        print("[agent] LLM policy ✓")
        return EmbeddingPolicy(
            sensitivity_threshold=data["sensitivity_threshold"], max_bpp=data["max_bpp"],
            allowed_data_types=data["allowed_data_types"], skip_roi=True,
            compliance=["HIPAA","GDPR"], rationale=data["rationale"],
            region_plan={}, llm_used=True,
        )
    except Exception as e:
        print(f"[agent] LLM failed ({e}) — rule-based fallback")
        return _rule_policy(modality, consent, sensitivity_map)

def plan_policy(modality, consent_level, sensitivity_map, use_llm=True):
    print(f"\n[agent] Policy  modality={modality}  consent={consent_level}")
    p = _llm_policy(modality, consent_level, sensitivity_map) if use_llm \
        else _rule_policy(modality, consent_level, sensitivity_map)
    print(f"[agent] Threshold={p.sensitivity_threshold}  Allowed={p.allowed_data_types}  LLM={p.llm_used}")
    return p

def enforce_compliance(policy, payload_type):
    ok = payload_type in policy.allowed_data_types
    print(f"[agent] Compliance '{payload_type}': {'ALLOWED ✓' if ok else 'BLOCKED ✗'}")
    return ok

def generate_audit_entry(policy, checksum, n_bits, ts):
    return {"timestamp":ts, "rationale":policy.rationale,
            "compliance":policy.compliance, "threshold":policy.sensitivity_threshold,
            "allowed":policy.allowed_data_types, "bits_embedded":n_bits,
            "checksum":checksum, "llm_assisted":policy.llm_used}
