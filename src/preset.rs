// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Timothy Redaelli

//! Preset loading and parameter injection.
//!
//! A [`Preset`] is built from one named section of a llama.cpp preset INI
//! file. It carries a stable **model identity string** (derived from the
//! `model`, `model-url`, `hf-repo`, and `hf-file` keys) and a map of
//! sampling parameters to inject into request bodies.
//!
//! [`build_presets`] is the main entry point; [`build_presets_from_sections`]
//! accepts already-parsed INI data and is used directly by unit tests.

use crate::ini::parse_ini_str;
use serde_json::{Map, Value};
use std::collections::HashMap;
use tracing::warn;

fn get_canonical_model_id_key(k: &str) -> Option<&'static str> {
    match k {
        "model" | "m" => Some("model"),
        "model-url" | "mu" => Some("model-url"),
        "hf-repo" | "hf" | "hfr" => Some("hf-repo"),
        "hf-file" | "hff" => Some("hf-file"),
        _ => None,
    }
}

fn get_sparam_json_key(k: &str) -> Option<&'static str> {
    match k {
        "temp" => Some("temperature"),
        "top-k" => Some("top_k"),
        "top-p" => Some("top_p"),
        "min-p" => Some("min_p"),
        "repeat-penalty" => Some("repeat_penalty"),
        "presence-penalty" => Some("presence_penalty"),
        "frequency-penalty" => Some("frequency_penalty"),
        "repeat-last-n" => Some("repeat_last_n"),
        "top-nsigma" => Some("top_n_sigma"),
        "xtc-probability" => Some("xtc_probability"),
        "xtc-threshold" => Some("xtc_threshold"),
        "typical" => Some("typical_p"),
        "dry-multiplier" => Some("dry_multiplier"),
        "dry-base" => Some("dry_base"),
        "dry-allowed-length" => Some("dry_allowed_length"),
        "dry-penalty-last-n" => Some("dry_penalty_last_n"),
        "mirostat" => Some("mirostat"),
        "mirostat-tau" => Some("mirostat_tau"),
        "mirostat-eta" => Some("mirostat_eta"),
        "seed" => Some("seed"),
        _ => None,
    }
}

fn is_truthy(v: &str) -> bool {
    matches!(v.to_lowercase().as_str(), "on" | "enabled" | "true" | "1")
}

fn is_autoy(v: &str) -> bool {
    matches!(v.to_lowercase().as_str(), "auto" | "-1")
}

fn compute_model_id(kv: &HashMap<String, String>) -> String {
    let mut canonical = HashMap::new();
    // Process entries in a deterministic (sorted-by-key) order. If a section sets
    // the same canonical key via two aliases with different values (e.g. both
    // `model` and `m`), last-writer-wins would otherwise depend on the HashMap's
    // randomized iteration order, making model_id — the routing/tracker key —
    // unstable across process runs.
    let mut pairs: Vec<(&String, &String)> = kv.iter().collect();
    pairs.sort_by(|a, b| a.0.cmp(b.0));
    for (key, val) in pairs {
        if let Some(canon) = get_canonical_model_id_key(key) {
            let trimmed = val.trim();
            if !trimmed.is_empty() {
                canonical.insert(canon, trimmed.to_string());
            }
        }
    }

    let mut parts = Vec::new();
    for canon in ["model", "model-url", "hf-repo", "hf-file"] {
        if let Some(val) = canonical.get(canon) {
            parts.push(format!("{}={}", canon, val));
        }
    }
    parts.join(";")
}

/// A single named preset loaded from an INI section.
#[derive(Debug, Clone)]
pub(crate) struct Preset {
    /// The section name as it appears in the INI file (used as the model alias).
    pub(crate) name: String,
    /// Stable identity string derived from the model-location keys (`model`,
    /// `model-url`, `hf-repo`, `hf-file`). Empty when none are present.
    pub(crate) model_id: String,
    /// Sampling parameters to inject into request bodies, keyed by their JSON
    /// field names (e.g. `"temperature"`, `"top_k"`).
    pub(crate) params: Map<String, Value>,
    /// Key-value pairs to merge into `chat_template_kwargs` (e.g.
    /// `enable_thinking` from the `reasoning` INI key).
    pub(crate) chat_template_kwargs: Map<String, Value>,
}

impl Preset {
    /// Inject this preset's parameters into a JSON request body.
    ///
    /// Only keys that the client did not already set are added. For
    /// `chat_template_kwargs`, individual sub-keys are merged with the same
    /// precedence rule: client values win.
    pub(crate) fn inject(&self, mut body: Map<String, Value>) -> Map<String, Value> {
        for (k, v) in &self.params {
            body.entry(k.clone()).or_insert_with(|| v.clone());
        }

        if !self.chat_template_kwargs.is_empty() {
            // Only merge if the client's value is an object or absent. If the client
            // sent a non-object, leave it untouched rather than silently overwriting.
            if body
                .get("chat_template_kwargs")
                .is_none_or(Value::is_object)
            {
                let mut ctk = body
                    .get("chat_template_kwargs")
                    .and_then(|v| v.as_object().cloned())
                    .unwrap_or_default();
                for (k, v) in &self.chat_template_kwargs {
                    ctk.entry(k.clone()).or_insert_with(|| v.clone());
                }
                body.insert("chat_template_kwargs".to_string(), Value::Object(ctk));
            }
        }
        body
    }
}

/// Build a preset map from the `data` array of a `/v1/models` JSON response.
///
/// Each entry's `status.preset` field is parsed as an INI snippet; non-default
/// sections are merged and forwarded to [`build_presets_from_sections`]. Entries
/// whose preset field is missing or fails to parse are skipped with a warning.
pub(crate) fn presets_from_models_json(data: &[Value]) -> HashMap<String, Preset> {
    let mut all_sections: HashMap<String, HashMap<String, String>> = HashMap::new();
    for entry in data {
        let id = entry.get("id").and_then(|v| v.as_str()).unwrap_or("?");
        let Some(preset_text) = entry["status"]["preset"].as_str() else {
            continue;
        };
        match parse_ini_str(preset_text) {
            Ok(sections) => {
                for (section, kv) in sections {
                    if section != "default" {
                        all_sections.entry(section).or_default().extend(kv);
                    }
                }
            }
            Err(e) => {
                warn!("failed to parse preset for {id:?}: {e}");
            }
        }
    }
    build_presets_from_sections(all_sections)
}

/// Build the preset map from already-parsed INI sections.
/// Extracted from `build_presets` to allow unit-testing without touching the filesystem.
pub(crate) fn build_presets_from_sections(
    mut sections: HashMap<String, HashMap<String, String>>,
) -> HashMap<String, Preset> {
    let global_kv = sections.remove("*").unwrap_or_default();

    let mut presets = HashMap::new();
    for (name, kv) in sections {
        // Skip reserved / invalid section names.
        if name.is_empty() || name.eq_ignore_ascii_case("default") {
            continue;
        }
        let mut merged = global_kv.clone();
        merged.extend(kv);

        let model_id = compute_model_id(&merged);
        let mut params = Map::new();
        let mut chat_template_kwargs = Map::new();

        for (key, val) in &merged {
            if let Some(json_key) = get_sparam_json_key(key) {
                let json_val = if let Ok(i) = val.parse::<i64>() {
                    Value::Number(i.into())
                } else if let Ok(u) = val.parse::<u64>() {
                    // Integers above i64::MAX (e.g. a full-range u64 seed) must stay
                    // exact; falling through to f64 would round and silently corrupt them.
                    Value::Number(u.into())
                } else if let Ok(f) = val.parse::<f64>() {
                    // from_f64 returns None for NaN/Infinity; fall back to string
                    match serde_json::Number::from_f64(f) {
                        Some(n) => Value::Number(n),
                        None => Value::String(val.clone()),
                    }
                } else {
                    Value::String(val.clone())
                };
                params.insert(json_key.to_string(), json_val);
            } else if key == "reasoning" && !is_autoy(val) {
                chat_template_kwargs
                    .insert("enable_thinking".to_string(), Value::Bool(is_truthy(val)));
            }
        }

        presets.insert(
            name.clone(),
            Preset {
                name,
                model_id,
                params,
                chat_template_kwargs,
            },
        );
    }
    presets
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn kv(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn sections(data: &[(&str, &[(&str, &str)])]) -> HashMap<String, HashMap<String, String>> {
        data.iter()
            .map(|(sec, pairs)| (sec.to_string(), kv(pairs)))
            .collect()
    }

    fn preset_with(params: &[(&str, Value)], ctk: &[(&str, Value)]) -> Preset {
        Preset {
            name: "test".to_string(),
            model_id: "model=test.gguf".to_string(),
            params: params
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect(),
            chat_template_kwargs: ctk
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect(),
        }
    }

    // -- compute_model_id ----------------------------------------------------

    #[test]
    fn model_id_empty_when_no_identity_keys() {
        assert_eq!(
            compute_model_id(&kv(&[("temp", "0.7"), ("top-k", "20")])),
            ""
        );
    }

    #[test]
    fn model_id_single_hf_repo() {
        assert_eq!(
            compute_model_id(&kv(&[("hf-repo", "org/model:Q4")])),
            "hf-repo=org/model:Q4"
        );
    }

    #[test]
    fn model_id_short_form_hf_equals_long_form() {
        let short = compute_model_id(&kv(&[("hf", "org/model:Q4")]));
        let long = compute_model_id(&kv(&[("hf-repo", "org/model:Q4")]));
        assert_eq!(short, long);
    }

    #[test]
    fn model_id_short_form_m_equals_long_form() {
        let short = compute_model_id(&kv(&[("m", "/path/model.gguf")]));
        let long = compute_model_id(&kv(&[("model", "/path/model.gguf")]));
        assert_eq!(short, long);
    }

    #[test]
    fn model_id_short_form_hfr_equals_hf_repo() {
        let a = compute_model_id(&kv(&[("hfr", "org/model:Q4")]));
        let b = compute_model_id(&kv(&[("hf-repo", "org/model:Q4")]));
        assert_eq!(a, b);
    }

    #[test]
    fn model_id_stable_ordering_independent_of_insertion() {
        // model always before model-url before hf-repo before hf-file
        let id = compute_model_id(&kv(&[("hf-file", "model.gguf"), ("hf-repo", "org/repo")]));
        assert_eq!(id, "hf-repo=org/repo;hf-file=model.gguf");
    }

    #[test]
    fn model_id_empty_value_ignored() {
        let id = compute_model_id(&kv(&[("hf-repo", ""), ("model", "test.gguf")]));
        assert_eq!(id, "model=test.gguf");
    }

    #[test]
    fn model_id_whitespace_trimmed() {
        let id = compute_model_id(&kv(&[("hf-repo", "  org/model:Q4  ")]));
        assert_eq!(id, "hf-repo=org/model:Q4");
    }

    // -- is_truthy / is_autoy ------------------------------------------------

    #[test]
    fn truthy_recognises_all_variants() {
        for v in &[
            "on", "ON", "On", "enabled", "ENABLED", "true", "TRUE", "True", "1",
        ] {
            assert!(is_truthy(v), "{v:?} should be truthy");
        }
    }

    #[test]
    fn truthy_rejects_non_truthy() {
        for v in &["off", "false", "0", "", "disabled", "no", "yes", "2"] {
            assert!(!is_truthy(v), "{v:?} should not be truthy");
        }
    }

    #[test]
    fn autoy_recognises_all_variants() {
        for v in &["auto", "AUTO", "Auto", "-1"] {
            assert!(is_autoy(v), "{v:?} should be autoy");
        }
    }

    #[test]
    fn autoy_rejects_non_autoy() {
        for v in &["on", "off", "0", "1", "true", "false", ""] {
            assert!(!is_autoy(v), "{v:?} should not be autoy");
        }
    }

    // -- Preset::inject ------------------------------------------------------

    #[test]
    fn inject_adds_missing_param() {
        let p = preset_with(&[("temperature", json!(0.7))], &[]);
        let body: Map<String, Value> = serde_json::from_str(r#"{"model":"x"}"#).unwrap();
        let result = p.inject(body);
        assert_eq!(result["temperature"], json!(0.7));
    }

    #[test]
    fn inject_does_not_overwrite_existing_param() {
        let p = preset_with(&[("temperature", json!(0.7))], &[]);
        let body: Map<String, Value> =
            serde_json::from_str(r#"{"model":"x","temperature":1.0}"#).unwrap();
        let result = p.inject(body);
        assert_eq!(result["temperature"], json!(1.0)); // client value wins
    }

    #[test]
    fn inject_merges_chat_template_kwargs_into_absent_field() {
        let p = preset_with(&[], &[("enable_thinking", json!(true))]);
        let body: Map<String, Value> = serde_json::from_str(r#"{"model":"x"}"#).unwrap();
        let result = p.inject(body);
        assert_eq!(
            result["chat_template_kwargs"]["enable_thinking"],
            json!(true)
        );
    }

    #[test]
    fn inject_merges_chat_template_kwargs_without_overwriting_existing_keys() {
        let p = preset_with(&[], &[("enable_thinking", json!(true))]);
        let body: Map<String, Value> = serde_json::from_str(
            r#"{"model":"x","chat_template_kwargs":{"enable_thinking":false}}"#,
        )
        .unwrap();
        let result = p.inject(body);
        assert_eq!(
            result["chat_template_kwargs"]["enable_thinking"],
            json!(false)
        );
    }

    #[test]
    fn inject_preserves_additional_ctk_keys_from_client() {
        let p = preset_with(&[], &[("enable_thinking", json!(true))]);
        let body: Map<String, Value> =
            serde_json::from_str(r#"{"model":"x","chat_template_kwargs":{"custom_key":"v"}}"#)
                .unwrap();
        let result = p.inject(body);
        assert_eq!(result["chat_template_kwargs"]["custom_key"], json!("v"));
        assert_eq!(
            result["chat_template_kwargs"]["enable_thinking"],
            json!(true)
        );
    }

    #[test]
    fn inject_preserves_non_object_chat_template_kwargs() {
        // If the client sends a non-object ctk, it must not be silently replaced.
        let p = preset_with(&[], &[("enable_thinking", json!(true))]);
        let body: Map<String, Value> =
            serde_json::from_str(r#"{"model":"x","chat_template_kwargs":"custom_string"}"#)
                .unwrap();
        let result = p.inject(body);
        assert_eq!(result["chat_template_kwargs"], json!("custom_string"));
    }

    #[test]
    fn inject_no_ctk_in_preset_leaves_client_ctk_untouched() {
        let p = preset_with(&[], &[]);
        let body: Map<String, Value> =
            serde_json::from_str(r#"{"model":"x","chat_template_kwargs":{"k":"v"}}"#).unwrap();
        let result = p.inject(body);
        assert_eq!(result["chat_template_kwargs"], json!({"k": "v"}));
    }

    #[test]
    fn inject_multiple_params_mixed_types() {
        let p = preset_with(
            &[
                ("temperature", json!(0.6)),
                ("top_k", json!(20_i64)),
                ("seed", json!(-1_i64)),
            ],
            &[],
        );
        let body: Map<String, Value> = serde_json::from_str(r#"{"model":"x"}"#).unwrap();
        let result = p.inject(body);
        assert_eq!(result["temperature"], json!(0.6));
        assert_eq!(result["top_k"], json!(20));
        assert_eq!(result["seed"], json!(-1));
    }

    // -- build_presets_from_sections -----------------------------------------

    #[test]
    fn build_global_merged_into_preset() {
        let s = sections(&[
            ("*", &[("temp", "0.7")]),
            ("alias", &[("hf-repo", "org/model:Q4")]),
        ]);
        let p = build_presets_from_sections(s);
        assert_eq!(p["alias"].params["temperature"], json!(0.7));
    }

    #[test]
    fn build_preset_overrides_global() {
        let s = sections(&[
            ("*", &[("temp", "0.7")]),
            ("alias", &[("hf-repo", "org/model:Q4"), ("temp", "0.9")]),
        ]);
        let p = build_presets_from_sections(s);
        assert_eq!(p["alias"].params["temperature"], json!(0.9));
    }

    #[test]
    fn build_default_section_skipped() {
        let s = sections(&[("DEFAULT", &[("temp", "0.5")])]);
        let p = build_presets_from_sections(s);
        assert!(!p.contains_key("DEFAULT"));
    }

    #[test]
    fn build_default_section_case_insensitive() {
        for name in &["default", "Default", "DEFAULT", "dEfAuLt"] {
            let s = sections(&[(name, &[("hf-repo", "org/m:Q4")])]);
            let p = build_presets_from_sections(s);
            assert!(p.is_empty(), "'{name}' section should be skipped");
        }
    }

    #[test]
    fn build_empty_section_name_skipped() {
        let mut s = sections(&[("alias", &[("hf-repo", "org/m:Q4")])]);
        s.insert("".to_string(), kv(&[("temp", "0.5")]));
        let p = build_presets_from_sections(s);
        assert!(!p.contains_key(""));
    }

    #[test]
    fn build_model_id_computed_from_hf_repo() {
        let s = sections(&[("alias", &[("hf-repo", "org/model:Q4_K_M")])]);
        let p = build_presets_from_sections(s);
        assert_eq!(p["alias"].model_id, "hf-repo=org/model:Q4_K_M");
    }

    #[test]
    fn build_no_identity_keys_yields_empty_model_id() {
        let s = sections(&[("alias", &[("temp", "0.7")])]);
        let p = build_presets_from_sections(s);
        assert_eq!(p["alias"].model_id, "");
    }

    #[test]
    fn build_reasoning_off_sets_enable_thinking_false() {
        let s = sections(&[("alias", &[("hf-repo", "org/m:Q4"), ("reasoning", "off")])]);
        let p = build_presets_from_sections(s);
        assert_eq!(
            p["alias"].chat_template_kwargs["enable_thinking"],
            json!(false)
        );
    }

    #[test]
    fn build_reasoning_on_sets_enable_thinking_true() {
        let s = sections(&[("alias", &[("hf-repo", "org/m:Q4"), ("reasoning", "on")])]);
        let p = build_presets_from_sections(s);
        assert_eq!(
            p["alias"].chat_template_kwargs["enable_thinking"],
            json!(true)
        );
    }

    #[test]
    fn build_reasoning_auto_leaves_ctk_empty() {
        let s = sections(&[("alias", &[("hf-repo", "org/m:Q4"), ("reasoning", "auto")])]);
        let p = build_presets_from_sections(s);
        assert!(p["alias"].chat_template_kwargs.is_empty());
    }

    // -- presets_from_models_json --------------------------------------------

    #[test]
    fn json_empty_data_yields_empty_map() {
        assert!(presets_from_models_json(&[]).is_empty());
    }

    #[test]
    fn json_entry_without_status_preset_is_skipped() {
        let data = vec![json!({"id": "alias-a"})];
        assert!(presets_from_models_json(&data).is_empty());
    }

    #[test]
    fn json_single_entry_builds_preset() {
        let data = vec![json!({
            "id": "alias-a",
            "status": {"preset": "[alias-a]\nhf-repo = org/model:Q4\ntemp = 0.7\n"}
        })];
        let p = presets_from_models_json(&data);
        assert!(p.contains_key("alias-a"));
        assert_eq!(p["alias-a"].model_id, "hf-repo=org/model:Q4");
        assert_eq!(p["alias-a"].params["temperature"], json!(0.7));
    }

    #[test]
    fn json_default_section_excluded() {
        let data = vec![json!({
            "id": "alias-a",
            "status": {"preset": "[default]\nhf-repo = org/model:Q4\n"}
        })];
        assert!(presets_from_models_json(&data).is_empty());
    }

    #[test]
    fn json_global_section_merged_into_preset() {
        let data = vec![json!({
            "id": "alias-a",
            "status": {"preset": "[*]\ntemp = 0.5\n[alias-a]\nhf-repo = org/model:Q4\n"}
        })];
        let p = presets_from_models_json(&data);
        assert_eq!(p["alias-a"].params["temperature"], json!(0.5));
    }

    #[test]
    fn json_two_entries_each_own_preset() {
        let data = vec![
            json!({"id": "alias-a", "status": {"preset": "[alias-a]\nhf-repo = org/model:Q4\ntemp = 0.6\n"}}),
            json!({"id": "alias-b", "status": {"preset": "[alias-b]\nhf-repo = org/model:Q4\ntemp = 0.9\n"}}),
        ];
        let p = presets_from_models_json(&data);
        assert_eq!(p.len(), 2);
        assert_eq!(p["alias-a"].params["temperature"], json!(0.6));
        assert_eq!(p["alias-b"].params["temperature"], json!(0.9));
    }

    // -- build_nan_float_falls_back_to_string (original test, kept in order) -

    #[test]
    fn build_nan_float_falls_back_to_string() {
        // "nan" parses as f64::NAN; from_f64 returns None → string fallback
        let s = sections(&[("alias", &[("hf-repo", "org/m:Q4"), ("temp", "nan")])]);
        let p = build_presets_from_sections(s);
        assert_eq!(p["alias"].params["temperature"], json!("nan"));
    }

    #[test]
    fn build_integer_param_stored_as_number() {
        let s = sections(&[("alias", &[("hf-repo", "org/m:Q4"), ("top-k", "20")])]);
        let p = build_presets_from_sections(s);
        assert_eq!(p["alias"].params["top_k"], json!(20));
    }

    #[test]
    fn build_two_aliases_same_model_id_differ_in_params() {
        let s = sections(&[
            ("alias-a", &[("hf-repo", "org/model:Q4"), ("temp", "0.6")]),
            ("alias-b", &[("hf-repo", "org/model:Q4"), ("temp", "0.9")]),
        ]);
        let p = build_presets_from_sections(s);
        assert_eq!(p["alias-a"].model_id, p["alias-b"].model_id);
        assert_ne!(
            p["alias-a"].params["temperature"],
            p["alias-b"].params["temperature"]
        );
    }

    #[test]
    fn build_u64_seed_kept_exact() {
        // A full-range u64 seed exceeds i64::MAX; it must stay an exact integer
        // rather than being rounded into an f64.
        let s = sections(&[(
            "alias",
            &[("hf-repo", "org/m:Q4"), ("seed", "18446744073709551615")],
        )]);
        let p = build_presets_from_sections(s);
        assert_eq!(p["alias"].params["seed"], json!(18446744073709551615_u64));
    }

    #[test]
    fn model_id_conflicting_aliases_deterministic() {
        // Both `model` and `m` map to the canonical "model" key. With deterministic
        // (sorted) key processing the long form wins, and the result is identical
        // across runs regardless of HashMap iteration order.
        for _ in 0..16 {
            let id = compute_model_id(&kv(&[("m", "B.gguf"), ("model", "A.gguf")]));
            assert_eq!(id, "model=A.gguf");
        }
    }
}
