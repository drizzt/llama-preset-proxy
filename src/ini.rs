// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Timothy Redaelli

//! INI file parser compatible with the llama.cpp preset format.
//!
//! The grammar is a direct translation of the C++ PEG parser in
//! `llama.cpp/common/preset.cpp`. Sections, key-value pairs, inline comments
//! (`; …` and `# …`), blank lines, and CRLF line endings are all supported.
//! Keys appearing before any section header land in the implicit `"default"`
//! section, matching the C++ parser's behaviour.

use pest::Parser as PestParser;
use std::collections::HashMap;

#[derive(pest_derive::Parser)]
#[grammar = "src/ini.pest"]
struct IniParser;

/// Parse an INI string using the pest grammar.
/// Keys before any section header go into the implicit "default" section,
/// matching the behaviour of the C++ PEG parser in llama.cpp/common/preset.cpp.
pub(crate) fn parse_ini_str(
    input: &str,
) -> Result<HashMap<String, HashMap<String, String>>, String> {
    let pairs =
        IniParser::parse(Rule::ini, input).map_err(|e| format!("INI parse error: {}", e))?;

    let mut sections: HashMap<String, HashMap<String, String>> = HashMap::new();
    let mut current_section = "default".to_string();
    sections.insert(current_section.clone(), HashMap::new());

    let ini_pair = pairs
        .into_iter()
        .next()
        .expect("pest must produce an ini node");
    for pair in ini_pair.into_inner() {
        match pair.as_rule() {
            Rule::header_line => {
                for inner in pair.into_inner() {
                    if inner.as_rule() == Rule::section_name {
                        // `section_name` is greedy up to `]`, so it captures any
                        // whitespace between the name and the closing bracket
                        // (`[ * ]` → "* "). Trim it so bracketed names with
                        // surrounding spaces match their bare form — otherwise
                        // "* "/"default " would silently bypass the global-merge
                        // and default-skip handling in build_presets_from_sections.
                        current_section = inner.as_str().trim().to_string();
                        sections.entry(current_section.clone()).or_default();
                    }
                }
            }
            Rule::kv_line => {
                let mut key = String::new();
                let mut val = String::new();
                for inner in pair.into_inner() {
                    match inner.as_rule() {
                        Rule::ident => key = inner.as_str().to_string(),
                        Rule::value => val = inner.as_str().to_string(),
                        _ => {}
                    }
                }
                sections
                    .entry(current_section.clone())
                    .or_default()
                    .insert(key, val);
            }
            _ => {}
        }
    }

    Ok(sections)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ini(content: &str) -> HashMap<String, HashMap<String, String>> {
        parse_ini_str(content).expect("parse_ini_str failed")
    }

    #[test]
    fn ini_basic_key_value() {
        let r = ini("[section]\nkey = value\n");
        assert_eq!(r["section"]["key"], "value");
    }

    #[test]
    fn ini_global_star_section() {
        let r = ini("[*]\ntemp = 0.7\n");
        assert_eq!(r["*"]["temp"], "0.7");
    }

    #[test]
    fn ini_multiple_sections() {
        let r = ini("[a]\nx = 1\n[b]\ny = 2\n");
        assert_eq!(r["a"]["x"], "1");
        assert_eq!(r["b"]["y"], "2");
    }

    #[test]
    fn ini_inline_semicolon_comment_stripped() {
        let r = ini("[s]\nkey = hello ; world\n");
        assert_eq!(r["s"]["key"], "hello");
    }

    #[test]
    fn ini_inline_hash_comment_stripped() {
        let r = ini("[s]\nkey = hello # world\n");
        assert_eq!(r["s"]["key"], "hello");
    }

    #[test]
    fn ini_full_line_comment_skipped() {
        let r = ini("[s]\n; this is a comment\nkey = val\n");
        assert_eq!(r["s"]["key"], "val");
        assert!(!r["s"].contains_key("; this is a comment"));
    }

    #[test]
    fn ini_equals_in_value() {
        let r = ini("[s]\nkey = a=b=c\n");
        assert_eq!(r["s"]["key"], "a=b=c");
    }

    #[test]
    fn ini_duplicate_keys_last_wins() {
        let r = ini("[s]\nkey = first\nkey = second\n");
        assert_eq!(r["s"]["key"], "second");
    }

    #[test]
    fn ini_keys_before_any_section_go_to_default() {
        // Keys before any section header land in the implicit "default" section,
        // matching the C++ parser which starts with current_section = "default".
        let r = ini("orphan = value\n[s]\nkey = val\n");
        assert_eq!(r["default"]["orphan"], "value");
        assert_eq!(r["s"]["key"], "val");
    }

    #[test]
    fn ini_section_name_with_special_chars() {
        let r = ini("[bartowski/model:Q4_K_M]\nx = 1\n");
        assert_eq!(r["bartowski/model:Q4_K_M"]["x"], "1");
    }

    #[test]
    fn ini_empty_section_name_is_error() {
        // [] has no section name (section_name requires ≥1 char), so it is a parse error.
        assert!(parse_ini_str("[real]\na = 1\n[]\nb = 2\n").is_err());
    }

    #[test]
    fn ini_section_header_with_no_closing_bracket_is_error() {
        // "[unclosed" has no ']', so header_line fails and no other line rule
        // matches, making the whole parse fail.
        assert!(parse_ini_str("[good]\na = 1\n[unclosed\nb = 2\n").is_err());
    }

    #[test]
    fn ini_empty_value() {
        let r = ini("[s]\nkey =\n");
        assert_eq!(r["s"]["key"], "");
    }

    #[test]
    fn ini_invalid_key_with_space_is_error() {
        // "bad key" contains a space; ident rule rejects it, causing a parse error.
        assert!(parse_ini_str("[s]\nbad key = val\n").is_err());
    }

    #[test]
    fn ini_crlf_line_endings() {
        let r = ini("[s]\r\nkey = val\r\n");
        assert_eq!(r["s"]["key"], "val");
    }

    #[test]
    fn ini_section_name_surrounding_whitespace_trimmed() {
        // `[ * ]` and `[ name ]` must normalise to "*"/"name", not "* "/"name ",
        // so the global-merge, default-skip, and alias routing keys match.
        let r = ini("[ * ]\ntemp = 0.5\n[ alias ]\nhf-repo = org/m:Q4\n");
        assert_eq!(r["*"]["temp"], "0.5");
        assert_eq!(r["alias"]["hf-repo"], "org/m:Q4");
        assert!(!r.contains_key("* "));
        assert!(!r.contains_key("alias "));
    }
}
