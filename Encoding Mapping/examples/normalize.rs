//! Normalize traditional Mongolian text from stdin to stdout.
//!
//! Usage:
//!   cargo run --example normalize -- [--nominal] < input.txt
//!
//! Flags:
//!   --nominal   collapse to nominal Unicode (recommended for tokenizer ingest)
//!   (default)   normalize to standard Unicode (preserves presentation variants)

use std::io::{self, Read, Write};

use encoding_mapping::{normalize_to_nominal_unicode, normalize_to_unicode};

fn main() {
    let mut nominal = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--nominal" => nominal = true,
            "--help" | "-h" => {
                eprintln!(
                    "Usage: normalize [--nominal] < input.txt\n\
                     Reads UTF-8 text from stdin and writes normalized text to stdout."
                );
                return;
            }
            other => {
                eprintln!("normalize: unknown argument {other:?}");
                std::process::exit(2);
            }
        }
    }

    let mut input = String::new();
    if let Err(e) = io::stdin().read_to_string(&mut input) {
        eprintln!("normalize: read error: {e}");
        std::process::exit(1);
    }

    let out = if nominal {
        normalize_to_nominal_unicode(&input)
    } else {
        normalize_to_unicode(&input)
    };

    if let Err(e) = io::stdout().write_all(out.as_bytes()) {
        eprintln!("normalize: write error: {e}");
        std::process::exit(1);
    }
}
